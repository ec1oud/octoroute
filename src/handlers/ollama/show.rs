//! Ollama-compatible model details handler
//!
//! Handles POST /api/show requests to return model information.
//! This endpoint is called by clients like Zed to verify model existence
//! and retrieve model configuration details.

use crate::config::ModelEndpoint;
use crate::handlers::AppState;
use axum::{Json, extract::State, response::IntoResponse};
use serde::{Deserialize, Serialize};

/// Request for model details
#[derive(Debug, Clone, Deserialize)]
pub struct ShowRequest {
    /// Name of the model to show information for
    pub model: String,
    /// Include verbose output (tensor information)
    #[serde(default)]
    pub verbose: bool,
}

/// Response with model details
#[derive(Debug, Clone, Serialize)]
pub struct ShowResponse {
    /// Modelfile content (synthetic)
    pub modelfile: String,
    /// Model parameters as text
    pub parameters: String,
    /// Prompt template
    pub template: String,
    /// System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// License information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Model architecture details
    pub details: ModelDetails,
    /// Low-level model metadata
    pub model_info: ModelInfo,
    /// Model capabilities
    pub capabilities: Vec<String>,
    /// Last modification timestamp
    pub modified_at: String,
}

/// Model architecture details
#[derive(Debug, Clone, Serialize)]
pub struct ModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

/// Model metadata from configuration
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    #[serde(rename = "general.architecture")]
    pub architecture: String,
    #[serde(rename = "general.file_type")]
    pub file_type: String,
    #[serde(rename = "general.parameter_count")]
    pub parameter_count: i64,
    #[serde(rename = "llama.context_length", skip_serializing_if = "Option::is_none")]
    pub llama_context_length: Option<i64>,
    #[serde(rename = "qwen2.context_length", skip_serializing_if = "Option::is_none")]
    pub qwen2_context_length: Option<i64>,
    #[serde(rename = "general.context_length")]
    pub context_length: i64,
}

/// POST /api/show handler
///
/// Returns model information in Ollama-compatible format.
/// Supports virtual model names (auto, fast, balanced, deep) and
/// specific endpoint names from configuration.
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<ShowRequest>,
) -> Result<impl IntoResponse, crate::error::AppError> {
    tracing::debug!(model = %request.model, "Received model show request");

    // Find the endpoint matching the model name
    let endpoint = find_endpoint(&state, &request.model)?;

    // Generate synthetic model information
    let response = build_show_response(&endpoint, &request.model);

    Ok(Json(response))
}

/// Find an endpoint by model name
fn find_endpoint(
    state: &AppState,
    model_name: &str,
) -> Result<ModelEndpoint, crate::error::AppError> {
    let config = state.config();

    // Check for virtual model names - use first endpoint from appropriate tier
    match model_name.to_lowercase().as_str() {
        "auto" | "fast" => {
            if let Some(endpoint) = config.models.fast.first() {
                return Ok(endpoint.clone());
            }
        }
        "balanced" => {
            if let Some(endpoint) = config.models.balanced.first() {
                return Ok(endpoint.clone());
            }
        }
        "deep" => {
            if let Some(endpoint) = config.models.deep.first() {
                return Ok(endpoint.clone());
            }
        }
        _ => {
            // Search for specific endpoint name in all tiers
            for endpoint in &config.models.fast {
                if endpoint.name() == model_name {
                    return Ok(endpoint.clone());
                }
            }
            for endpoint in &config.models.balanced {
                if endpoint.name() == model_name {
                    return Ok(endpoint.clone());
                }
            }
            for endpoint in &config.models.deep {
                if endpoint.name() == model_name {
                    return Ok(endpoint.clone());
                }
            }
        }
    }

    // If no specific match found, return first available endpoint
    if let Some(endpoint) = config.models.fast.first() {
        return Ok(endpoint.clone());
    }
    if let Some(endpoint) = config.models.balanced.first() {
        return Ok(endpoint.clone());
    }
    if let Some(endpoint) = config.models.deep.first() {
        return Ok(endpoint.clone());
    }

    Err(crate::error::AppError::Validation(format!(
        "Model '{}' not found",
        model_name
    )))
}

/// Build the show response from endpoint configuration
fn build_show_response(endpoint: &ModelEndpoint, requested_name: &str) -> ShowResponse {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Infer model family and size from name
    let (family, parameter_size) = infer_model_info(requested_name);

    // Extract context length from endpoint config
    let max_tokens = endpoint.max_tokens() as i64;

    ShowResponse {
        modelfile: format!(
            "# Modelfile for {}\n# Served via Octoroute\n\nFROM unknown\n",
            requested_name
        ),
        parameters: format!(
            "temperature {}\nmax_tokens {}\n",
            endpoint.temperature(),
            max_tokens
        ),
        template: "{{ .Prompt }}".to_string(),
        system: Some("You are a helpful assistant.".to_string()),
        license: Some("Unknown license - see model documentation".to_string()),
        details: ModelDetails {
            parent_model: String::new(),
            format: "gguf".to_string(),
            family: family.clone(),
            families: vec![family.clone()],
            parameter_size: parameter_size.clone(),
            quantization_level: "unknown".to_string(),
        },
        model_info: ModelInfo {
            architecture: family.clone(),
            file_type: "Q4_K_M".to_string(),
            parameter_count: parse_parameter_size(&parameter_size),
            llama_context_length: if family == "llama" {
                Some(max_tokens)
            } else {
                None
            },
            qwen2_context_length: if family == "qwen2" {
                Some(max_tokens)
            } else {
                None
            },
            context_length: max_tokens,
        },
        capabilities: vec!["completion".to_string()],
        modified_at: format!("{}-01-01T00:00:00Z", now),
    }
}

/// Infer model family and parameter size from model name
fn infer_model_info(name: &str) -> (String, String) {
    let lower = name.to_lowercase();

    // Try to detect family
    let family = if lower.contains("llama") {
        "llama"
    } else if lower.contains("qwen") {
        "qwen2"
    } else if lower.contains("gemma") {
        "gemma"
    } else if lower.contains("mistral") || lower.contains("mixtral") {
        "llama"
    } else if lower.contains("phi") {
        "phi"
    } else {
        "unknown"
    };

    // Try to extract parameter size
    let size = if lower.contains("70b") {
        "70B"
    } else if lower.contains("32b") {
        "32B"
    } else if lower.contains("14b") || lower.contains("15b") {
        "15B"
    } else if lower.contains("8b") {
        "8B"
    } else if lower.contains("7b") {
        "7B"
    } else if lower.contains("4b") || lower.contains("3b") {
        "3B"
    } else if lower.contains("2b") {
        "2B"
    } else if lower.contains("0.5b") || lower.contains("1b") {
        "1B"
    } else {
        "unknown"
    };

    (family.to_string(), size.to_string())
}

/// Parse parameter size string to approximate count
fn parse_parameter_size(size: &str) -> i64 {
    match size {
        "70B" => 70_000_000_000,
        "32B" => 32_000_000_000,
        "15B" => 15_000_000_000,
        "8B" => 8_000_000_000,
        "7B" => 7_000_000_000,
        "3B" => 3_000_000_000,
        "2B" => 2_000_000_000,
        "1B" => 1_000_000_000,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_model_info() {
        let (family, size) = infer_model_info("llama3.2-8b");
        assert_eq!(family, "llama");
        assert_eq!(size, "8B");

        let (family, size) = infer_model_info("qwen3-32b");
        assert_eq!(family, "qwen2");
        assert_eq!(size, "32B");

        let (family, size) = infer_model_info("gemma-7b");
        assert_eq!(family, "gemma");
        assert_eq!(size, "7B");
    }

    #[test]
    fn test_parse_parameter_size() {
        assert_eq!(parse_parameter_size("8B"), 8_000_000_000);
        assert_eq!(parse_parameter_size("unknown"), 0);
    }
}
