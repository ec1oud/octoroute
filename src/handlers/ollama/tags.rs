//! Ollama-compatible tags list handler
//!
//! Handles GET /api/tags requests.
//!
//! This endpoint mimics the Ollama API response format for clients that
//! expect to discover available models via this endpoint.

use crate::handlers::AppState;
use axum::{Json, extract::State, response::IntoResponse};

/// Ollama-compatible tags response
#[derive(Debug, Clone, serde::Serialize)]
pub struct TagsListResponse {
    pub models: Vec<ApiTagModel>,
}

/// A model object in Ollama tags format
#[derive(Debug, Clone, serde::Serialize)]
pub struct ApiTagModel {
    pub name: String,
    pub digest: String,
    pub size: u64,
    pub format: String,
    pub family: Option<String>,
    pub families: Option<Vec<String>>,
    pub parameter_size: String,
    pub quantization_level: String,
}

impl ApiTagModel {
    /// Create a new API tag model
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        // Generate a fake digest (in practice, this would be a hash)
        let digest = format!("sha256:{}", hex_digest(&name));

        Self {
            name,
            digest,
            size: 0,                    // Not available for Octoroute configured models
            format: "gguf".to_string(), // Default format
            family: None,               // Not tracked
            families: None,             // Not tracked
            parameter_size: "unknown".to_string(), // Not tracked
            quantization_level: "unknown".to_string(), // Not tracked
        }
    }
}

/// Generate a fake deterministic digest for a model name
/// This is just for compatibility - we don't actually track digests
/// TODO in the health check, parse the JSON and cache the hash
/// so we can return it when answering an /api/tags request
fn hex_digest(s: &str) -> String {
    // Use a simple hash for deterministic but unique-looking digests
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    let hash = hasher.finish();
    format!("{:016x}", hash)
}

/// GET /api/tags handler
///
/// Returns a list of available models in Ollama-compatible format.
///
/// # Response Format
///
/// Returns an object with:
/// - `models`: Array of model objects, each containing:
///   - `name`: Model name
///   - `digest`: Fake SHA256 digest (for compatibility)
///   - `size`: Size in bytes (reported as 0 for Octoroute models)
///   - `format`: Model format (default: "gguf")
///   - `family`: Model family (not tracked, omitted)
///   - `families`: Model families (not tracked, omitted)
///   - `parameter_size`: Parameter size (reported as "unknown")
///   - `quantization_level`: Quantization level (reported as "unknown")
///
/// The response includes:
/// - Tier-based virtual models: `auto`, `fast`, `balanced`, `deep`
/// - All configured endpoint names from the config file
pub async fn handler(State(state): State<AppState>) -> impl IntoResponse {
    let mut models = Vec::new();

    // Add tier-based virtual models
    for tier in ["auto", "fast", "balanced", "deep"] {
        models.push(ApiTagModel::new(tier));
    }

    // Add configured endpoint names from each tier
    for endpoint in &state.config().models.fast {
        models.push(ApiTagModel::new(endpoint.name()));
    }
    for endpoint in &state.config().models.balanced {
        models.push(ApiTagModel::new(endpoint.name()));
    }
    for endpoint in &state.config().models.deep {
        models.push(ApiTagModel::new(endpoint.name()));
    }

    Json(TagsListResponse { models })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_tag_model_creation() {
        let model = ApiTagModel::new("qwen3-8b");
        assert_eq!(model.name, "qwen3-8b");
        assert!(model.digest.starts_with("sha256:"));
        assert_eq!(model.format, "gguf");
        assert_eq!(model.parameter_size, "unknown");
    }

    #[test]
    fn test_tags_list_response() {
        let models = vec![ApiTagModel::new("auto"), ApiTagModel::new("qwen3-8b")];
        let response = TagsListResponse { models };
        assert_eq!(response.models.len(), 2);
    }

    #[test]
    fn test_hex_digest_deterministic() {
        let digest1 = hex_digest("test-model");
        let digest2 = hex_digest("test-model");
        assert_eq!(digest1, digest2);

        let digest3 = hex_digest("different-model");
        assert_ne!(digest1, digest3);
    }
}
