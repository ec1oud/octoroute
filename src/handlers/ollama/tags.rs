//! Ollama-compatible tags list handler
//!
//! Handles GET /api/tags requests.
//!
//! This endpoint mimics the Ollama API response format for clients that
//! expect to discover available models via this endpoint.

use crate::handlers::AppState;
use crate::models::cache::ModelInfo;
use axum::{Json, extract::State, response::IntoResponse};

/// Ollama-compatible tags response
#[derive(Debug, Clone, serde::Serialize)]
pub struct TagsListResponse {
    pub models: Vec<TagsModel>,
}

/// A model in the /api/tags response format
#[derive(Debug, Clone, serde::Serialize)]
pub struct TagsModel {
    pub name: String,
    #[serde(rename = "model")]
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<TagsModelDetails>,
}

/// Details object in the /api/tags response
#[derive(Debug, Clone, serde::Serialize)]
pub struct TagsModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
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
///   - `model`: Duplicate of name (Ollama API requirement)
///   - `modified_at`: Last modification timestamp
///   - `digest`: SHA256 digest
///   - `size`: Size in bytes (0 for Octoroute virtual models)
///   - `details`: Object with model metadata (present for local models)
///     - `parent_model`: Parent model name
///     - `format`: Model format (e.g., "gguf")
///     - `family`: Model family name
///     - `families`: List of family names
///     - `parameter_size`: Parameter count (e.g., "31.6B")
///     - `quantization_level`: Quantization (e.g., "Q4_K_M")
///
/// The response includes:
/// - Tier-based virtual models: `auto`, `fast`, `balanced`, `deep`
/// - All configured endpoint names from the config file
pub async fn handler(State(state): State<AppState>) -> impl IntoResponse {
    // Get the model cache to look up full model info
    let model_cache = state.model_cache();
    let model_cache_guard = model_cache.read().await;

    let models = {
        // Scope block to ensure the guard lives for the entire model population
        let mut models = Vec::new();

        // Add tier-based virtual models (no cached info, use placeholder)
        for tier in ["auto", "fast", "balanced", "deep"] {
            models.push(TagsModel {
                name: tier.to_string(),
                model: tier.to_string(),
                modified_at: "2026-01-01T00:00:00Z".to_string(),
                size: 0,
                digest: format!("sha256:{:016x}", hash_string(tier)),
                details: None,
            });
        }

        // Helper to convert cached ModelInfo to TagsModel
        let model_to_tags = |info: &ModelInfo| -> TagsModel {
            TagsModel {
                name: info.name.clone(),
                model: info.model.clone(),
                modified_at: info.modified_at.clone(),
                size: info.size,
                digest: info.digest.clone(),
                details: info.details.as_ref().map(|d| TagsModelDetails {
                    parent_model: d.parent_model.clone(),
                    format: d.format.clone(),
                    family: d.family.clone(),
                    families: d.families.clone(),
                    parameter_size: d.parameter_size.clone(),
                    quantization_level: d.quantization_level.clone(),
                }),
            }
        };

        // Add configured endpoint names from each tier
        for endpoint in &state.config().models.fast {
            if let Some(info) = model_cache_guard.get(endpoint.name()) {
                models.push(model_to_tags(info));
            }
        }
        for endpoint in &state.config().models.balanced {
            if let Some(info) = model_cache_guard.get(endpoint.name()) {
                models.push(model_to_tags(info));
            }
        }
        for endpoint in &state.config().models.deep {
            if let Some(info) = model_cache_guard.get(endpoint.name()) {
                models.push(model_to_tags(info));
            }
        }

        models
    };

    Json(TagsListResponse { models })
}

/// Generate a deterministic hash string for a model name
/// Used for virtual models that don't have real digests
fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
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
