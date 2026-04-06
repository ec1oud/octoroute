//! Ollama-compatible generate endpoint handler
//!
//! Handles POST /api/generate requests for text completion (non-chat).
//! Some clients prefer this over /api/chat for simpler use cases.

use crate::error::{AppError, AppResult};
use crate::handlers::AppState;
use crate::middleware::RequestId;
use crate::router::{Importance, RouteMetadata, RoutingStrategy, TargetModel, TaskType};
use crate::shared::query::{QueryConfig, execute_query_with_retry};
use axum::{Extension, Json, extract::State, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Ollama-compatible generate request
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GenerateRequest {
    /// Model name to use (can be "auto", tier name, or specific endpoint name)
    pub model: String,
    /// The prompt to complete
    pub prompt: String,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// System prompt override
    pub system: Option<String>,
    /// Model options (temperature, max tokens, etc.)
    #[serde(default)]
    pub options: GenerateOptions,
    /// Context from previous generation (for maintaining state)
    pub context: Option<Vec<i32>>,
}

/// Model options for Ollama generate
#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct GenerateOptions {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default, rename = "num_predict")]
    pub max_tokens: Option<i32>,
    #[serde(default)]
    pub num_ctx: Option<i32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
}

/// Ollama-compatible generate response
#[derive(Debug, Clone, Serialize)]
pub struct GenerateResponse {
    pub model: String,
    #[serde(rename = "created_at")]
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

impl GenerateResponse {
    /// Create a new generate response
    pub fn new(model: String, response_text: String, timestamp: i64) -> Self {
        Self {
            model,
            created_at: timestamp.to_string(),
            response: response_text,
            done: true,
            context: None,
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
        }
    }
}

/// Build RouteMetadata from generate request for routing
fn build_route_metadata(request: &GenerateRequest, token_estimate: usize) -> RouteMetadata {
    let metadata = RouteMetadata::new(token_estimate);

    // Infer importance from prompt content
    let content_lower = request.prompt.to_lowercase();

    let importance = if content_lower.contains("urgent")
        || content_lower.contains("critical")
        || content_lower.contains("asap")
    {
        Importance::High
    } else if content_lower.contains("casual")
        || content_lower.contains("quick")
        || content_lower.contains("simple")
    {
        Importance::Low
    } else {
        Importance::Normal
    };

    // Infer task type from content
    let task_type = if content_lower.contains("code")
        || content_lower.contains("function")
        || content_lower.contains("implement")
        || content_lower.contains("class")
    {
        TaskType::Code
    } else if content_lower.contains("explain")
        || content_lower.contains("describe")
        || content_lower.contains("what is")
        || content_lower.contains("how to")
    {
        TaskType::QuestionAnswer
    } else if content_lower.contains("creative")
        || content_lower.contains("write")
        || content_lower.contains("compose")
        || content_lower.contains("story")
    {
        TaskType::CreativeWriting
    } else {
        TaskType::CasualChat
    };

    metadata.with_importance(importance).with_task_type(task_type)
}

/// Parse model choice to determine target
/// Returns None if model is "auto" (use routing), or the target tier
fn parse_model_choice(model: &str) -> AppResult<Option<TargetModel>> {
    match model.to_lowercase().as_str() {
        "auto" => Ok(None), // Use intelligent routing
        "fast" => Ok(Some(TargetModel::Fast)),
        "balanced" => Ok(Some(TargetModel::Balanced)),
        "deep" => Ok(Some(TargetModel::Deep)),
        // For other model names, use balanced tier as default
        _ => {
            // Model name lookup could be added later if needed
            Ok(Some(TargetModel::Balanced))
        }
    }
}

/// POST /api/generate handler
///
/// Handles text completion requests in Ollama's native format.
pub async fn handler(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    Json(request): Json<GenerateRequest>,
) -> Result<impl IntoResponse, AppError> {
    tracing::debug!(
        request_id = %request_id,
        model = %request.model,
        prompt_length = request.prompt.len(),
        stream = request.stream,
        "Received Ollama generate request"
    );

    // Validate prompt is not empty
    if request.prompt.trim().is_empty() {
        return Err(AppError::Validation("Prompt cannot be empty".to_string()));
    }

    let token_estimate = request.prompt.len() / 4;
    let metadata = build_route_metadata(&request, token_estimate);

    // Parse model choice
    let parsed = parse_model_choice(&request.model)?;

    // Determine target model
    let decision = if let Some(target) = parsed {
        // Use the selected tier directly
        crate::router::RoutingDecision::new(target, RoutingStrategy::Rule)
    } else {
        // Use intelligent routing
        let routing_start = std::time::Instant::now();
        let decision = state
            .router()
            .route(&request.prompt, &metadata, state.selector())
            .await?;
        let duration_ms = routing_start.elapsed().as_secs_f64() * 1000.0;

        tracing::info!(
            request_id = %request_id,
            target_tier = ?decision.target(),
            routing_strategy = ?decision.strategy(),
            token_estimate = metadata.token_estimate,
            routing_duration_ms = %duration_ms,
            "Ollama generate routing decision made"
        );
        decision
    };

    // Build the full prompt (system + user prompt)
    let full_prompt = if let Some(system) = &request.system {
        format!("{}\n\n{}", system, request.prompt)
    } else {
        request.prompt.clone()
    };

    // Execute query
    let config = QueryConfig::default();
    let result = execute_query_with_retry(
        &state,
        &decision,
        &full_prompt,
        request_id,
        &config,
        None,
    )
    .await?;

    // Create timestamp
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_secs() as i64;

    // Build Ollama response
    let response = GenerateResponse::new(
        result.endpoint.name().to_string(),
        result.content,
        timestamp,
    );

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_response_creation() {
        let response =
            GenerateResponse::new("test-model".to_string(), "Hello!".to_string(), 1234567890);
        assert_eq!(response.model, "test-model");
        assert_eq!(response.response, "Hello!");
        assert!(response.done);
        assert_eq!(response.created_at, "1234567890");
    }

    #[test]
    fn test_build_route_metadata_from_code() {
        let request = GenerateRequest {
            model: "auto".to_string(),
            prompt: "Write a function to sort a list".to_string(),
            stream: false,
            system: None,
            options: Default::default(),
            context: None,
        };

        let metadata = build_route_metadata(&request, 100);
        assert_eq!(metadata.task_type, TaskType::Code);
    }

    #[test]
    fn test_build_route_metadata_from_urgent() {
        let request = GenerateRequest {
            model: "auto".to_string(),
            prompt: "Urgent: critical bug fix needed".to_string(),
            stream: false,
            system: None,
            options: Default::default(),
            context: None,
        };

        let metadata = build_route_metadata(&request, 100);
        assert_eq!(metadata.importance, Importance::High);
    }

    #[test]
    fn test_parse_model_choice() {
        assert_eq!(parse_model_choice("auto").unwrap(), None);
        assert_eq!(
            parse_model_choice("fast").unwrap(),
            Some(TargetModel::Fast)
        );
        assert_eq!(
            parse_model_choice("balanced").unwrap(),
            Some(TargetModel::Balanced)
        );
        assert_eq!(
            parse_model_choice("deep").unwrap(),
            Some(TargetModel::Deep)
        );
        assert_eq!(
            parse_model_choice("custom-model").unwrap(),
            Some(TargetModel::Balanced)
        ); // defaults to balanced
    }
}
