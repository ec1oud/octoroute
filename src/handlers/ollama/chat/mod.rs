//! Ollama-compatible chat endpoint handler
//!
//! Handles POST /api/chat requests in Ollama's native format.
//! This enables tools that expect to talk directly to Ollama (like Zed editor)
//! to work seamlessly through Octoroute's intelligent routing.

use crate::error::{AppError, AppResult};
use crate::handlers::AppState;
use crate::middleware::RequestId;
use crate::router::{Importance, RouteMetadata, RoutingStrategy, TargetModel, TaskType};
use crate::shared::query::{QueryConfig, execute_query_with_retry};
use axum::{Extension, Json, extract::State, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Ollama-compatible chat request
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct OllamaChatRequest {
    /// Model name to use (can be "auto", tier name, or specific endpoint name)
    pub model: String,
    /// Messages in the conversation
    pub messages: Vec<OllamaMessage>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Model options (temperature, max tokens, etc.)
    #[serde(default)]
    pub options: OllamaOptions,
    /// System prompt override
    pub system: Option<String>,
}

/// A message in an Ollama chat request
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct OllamaMessage {
    pub role: OllamaRole,
    pub content: String,
}

/// Message role (user, system, assistant)
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OllamaRole {
    User,
    System,
    Assistant,
}

/// Model options for Ollama chat
#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct OllamaOptions {
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

/// Ollama-compatible chat response (non-streaming)
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    #[serde(rename = "created_at")]
    pub created_at: String,
    pub message: OllamaMessageResponse,
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

/// Response message content
#[derive(Debug, Clone, Serialize)]
pub struct OllamaMessageResponse {
    pub role: OllamaRole,
    pub content: String,
}

impl Serialize for OllamaRole {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            OllamaRole::User => "user",
            OllamaRole::System => "system",
            OllamaRole::Assistant => "assistant",
        };
        variant.serialize(serializer)
    }
}

impl OllamaChatResponse {
    /// Create a new chat response
    pub fn new(model: String, content: String, timestamp: i64) -> Self {
        Self {
            model,
            created_at: timestamp.to_string(),
            message: OllamaMessageResponse {
                role: OllamaRole::Assistant,
                content,
            },
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

/// Extract the user message content from Ollama messages
/// This handles both single messages and conversation history
fn extract_user_message(messages: &[OllamaMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|msg| matches!(msg.role, OllamaRole::User))
        .map(|msg| msg.content.clone())
}

/// Build RouteMetadata from Ollama request for routing
fn build_route_metadata(request: &OllamaChatRequest, token_estimate: usize) -> RouteMetadata {
    let metadata = RouteMetadata::new(token_estimate);

    // Infer importance from message content
    let content_lower = request
        .messages
        .iter()
        .map(|m| m.content.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    // Simple heuristics for importance based on content
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
    {
        TaskType::Code
    } else if content_lower.contains("explain")
        || content_lower.contains("describe")
        || content_lower.contains("what is")
    {
        TaskType::QuestionAnswer
    } else if content_lower.contains("creative")
        || content_lower.contains("write")
        || content_lower.contains("compose")
    {
        TaskType::CreativeWriting
    } else {
        TaskType::CasualChat
    };

    metadata
        .with_importance(importance)
        .with_task_type(task_type)
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

/// POST /api/chat handler (non-streaming, and also streaming via SSE fallback)
pub async fn handler(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    Json(request): Json<OllamaChatRequest>,
) -> Result<impl IntoResponse, AppError> {
    tracing::debug!(
        request_id = %request_id,
        model = %request.model,
        message_count = request.messages.len(),
        stream = request.stream.unwrap_or(false),
        "Received Ollama chat request"
    );

    // Extract user message
    let user_message = match extract_user_message(&request.messages) {
        Some(msg) => msg,
        None => {
            return Err(AppError::Validation(
                "No user message found in conversation".to_string(),
            ));
        }
    };

    let token_estimate = user_message.len() / 4;
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
            .route(&user_message, &metadata, state.selector())
            .await?;
        let duration_ms = routing_start.elapsed().as_secs_f64() * 1000.0;

        tracing::info!(
            request_id = %request_id,
            target_tier = ?decision.target(),
            routing_strategy = ?decision.strategy(),
            token_estimate = metadata.token_estimate,
            routing_duration_ms = %duration_ms,
            "Ollama chat routing decision made"
        );
        decision
    };

    // Check if streaming is requested - for now, we'll do non-streaming only
    // Streaming can be added later with proper async stream handling
    let _streaming = request.stream.unwrap_or(false);

    // Execute query
    let config = QueryConfig::default();
    let result =
        execute_query_with_retry(&state, &decision, &user_message, request_id, &config, None)
            .await?;

    // Create timestamp
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_secs() as i64;

    // Build Ollama response
    let response = OllamaChatResponse::new(
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
    fn test_extract_user_message() {
        let messages = vec![
            OllamaMessage {
                role: OllamaRole::System,
                content: "You are helpful".to_string(),
            },
            OllamaMessage {
                role: OllamaRole::User,
                content: "Hello".to_string(),
            },
            OllamaMessage {
                role: OllamaRole::Assistant,
                content: "Hi there!".to_string(),
            },
            OllamaMessage {
                role: OllamaRole::User,
                content: "How are you?".to_string(),
            },
        ];

        assert_eq!(
            extract_user_message(&messages),
            Some("How are you?".to_string())
        );
    }

    #[test]
    fn test_extract_user_message_only_user() {
        let messages = vec![OllamaMessage {
            role: OllamaRole::User,
            content: "Hello".to_string(),
        }];

        assert_eq!(extract_user_message(&messages), Some("Hello".to_string()));
    }

    #[test]
    fn test_extract_user_message_no_user() {
        let messages = vec![
            OllamaMessage {
                role: OllamaRole::System,
                content: "You are helpful".to_string(),
            },
            OllamaMessage {
                role: OllamaRole::Assistant,
                content: "Hi there!".to_string(),
            },
        ];

        assert_eq!(extract_user_message(&messages), None);
    }

    #[test]
    fn test_ollama_chat_response() {
        let response =
            OllamaChatResponse::new("test-model".to_string(), "Hello!".to_string(), 1234567890);
        assert_eq!(response.model, "test-model");
        assert_eq!(response.message.content, "Hello!");
        assert!(response.done);
        assert_eq!(response.created_at, "1234567890");
    }

    #[test]
    fn test_build_route_metadata_from_casual() {
        let request = OllamaChatRequest {
            model: "auto".to_string(),
            messages: vec![OllamaMessage {
                role: OllamaRole::User,
                content: "quick question".to_string(),
            }],
            stream: Some(false),
            options: Default::default(),
            system: None,
        };

        let metadata = build_route_metadata(&request, 100);
        assert_eq!(metadata.importance, Importance::Low);
    }

    #[test]
    fn test_build_route_metadata_from_urgent() {
        let request = OllamaChatRequest {
            model: "auto".to_string(),
            messages: vec![OllamaMessage {
                role: OllamaRole::User,
                content: "urgent critical question".to_string(),
            }],
            stream: Some(false),
            options: Default::default(),
            system: None,
        };

        let metadata = build_route_metadata(&request, 100);
        assert_eq!(metadata.importance, Importance::High);
    }
}
