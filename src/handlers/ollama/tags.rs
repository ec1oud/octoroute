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
    /// Create a new API tag model with optional cached digest
    pub fn new(name: impl Into<String>, cached_digest: Option<&str>) -> Self {
        let name = name.into();
        // Use cached real digest if available, otherwise generate a fake one
        let digest = cached_digest
            .map(|d| d.to_string())
            .unwrap_or_else(|| format!("sha256:{}", hex_digest(&name)));

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
/// This is used as a fallback when no real digest is cached
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
    // Get the hash cache to look up real digests
    let hash_cache = state.hash_cache();
    let hash_cache_guard = hash_cache.read().await;

    let models = {
        // Scope block to ensure the guard lives for the entire model population
        let mut models = Vec::new();

        // Get cached digest helper - closure captures hash_cache_guard
        let get_digest = |name: &str| hash_cache_guard.get(name).map(|s| s.as_str());

        // Add tier-based virtual models
        for tier in ["auto", "fast", "balanced", "deep"] {
            models.push(ApiTagModel::new(tier, get_digest(tier)));
        }

        // Add configured endpoint names from each tier
        for endpoint in &state.config().models.fast {
            models.push(ApiTagModel::new(
                endpoint.name(),
                get_digest(endpoint.name()),
            ));
        }
        for endpoint in &state.config().models.balanced {
            models.push(ApiTagModel::new(
                endpoint.name(),
                get_digest(endpoint.name()),
            ));
        }
        for endpoint in &state.config().models.deep {
            models.push(ApiTagModel::new(
                endpoint.name(),
                get_digest(endpoint.name()),
            ));
        }

        models
    };

    Json(TagsListResponse { models })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_tag_model_creation() {
        let model = ApiTagModel::new("qwen3-8b", None);
        assert_eq!(model.name, "qwen3-8b");
        assert!(model.digest.starts_with("sha256:"));
        assert_eq!(model.format, "gguf");
        assert_eq!(model.parameter_size, "unknown");
    }

    #[test]
    fn test_api_tag_model_with_cached_digest() {
        let cached = "sha256:abc123def456";
        let model = ApiTagModel::new("qwen3-8b", Some(cached));
        assert_eq!(model.name, "qwen3-8b");
        assert_eq!(model.digest, cached);
    }

    #[test]
    fn test_tags_list_response() {
        let models = vec![
            ApiTagModel::new("auto", None),
            ApiTagModel::new("qwen3-8b", None),
        ];
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
