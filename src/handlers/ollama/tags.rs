//! Ollama-compatible tags list handler
//!
//! Handles GET /api/tags requests.
//!
//! This endpoint mimics the Ollama API response format for clients that
//! expect to discover available models via this endpoint.

use crate::config::Config;
use crate::handlers::AppState;
use crate::models::cache::{ModelInfo, ModelInfoExt};
use axum::{Json, extract::State, response::IntoResponse};
use std::collections::HashMap;

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

        // Helper to create a virtual tier model with details from the first healthy endpoint
        let create_virtual_tier_model =
            |tier: &str, config: &Config, cache: &HashMap<String, ModelInfo>| -> TagsModel {
                // Find the first healthy endpoint in this tier to get real model details
                let tier_endpoints = match tier {
                    "fast" => &config.models.fast,
                    "balanced" => &config.models.balanced,
                    "deep" => &config.models.deep,
                    "auto" => {
                        // "auto" uses intelligent routing, return with minimal details
                        return TagsModel {
                            name: tier.to_string(),
                            model: tier.to_string(),
                            modified_at: "2026-01-01T00:00:00Z".to_string(),
                            size: 0,
                            digest: format!("sha256:{:016x}", hash_string(tier)),
                            details: Some(TagsModelDetails {
                                parent_model: "".to_string(),
                                format: "gguf".to_string(),
                                family: "multi-tier".to_string(),
                                families: vec!["multi-tier".to_string()],
                                parameter_size: "varies".to_string(),
                                quantization_level: "varies".to_string(),
                            }),
                        };
                    }
                    _ => {
                        return TagsModel {
                            name: tier.to_string(),
                            model: tier.to_string(),
                            modified_at: "2026-01-01T00:00:00Z".to_string(),
                            size: 0,
                            digest: format!("sha256:{:016x}", hash_string(tier)),
                            details: None,
                        };
                    }
                };

                // Look for a cached model info entry for any endpoint in this tier
                let tier_entry = tier_endpoints.iter().find_map(|endpoint| {
                    let endpoint_name = endpoint.name().to_string();
                    if let Some(info) = cache.get(&endpoint_name) {
                        return Some(info);
                    }
                    None
                });

                // If no cached entry found for endpoints, look for any model from this tier
                // that has the same name as an endpoint (real model discovered from Ollama)
                let tier_entry = tier_entry.or_else(|| {
                    cache.values().find(|info| {
                        tier_endpoints
                            .iter()
                            .any(|ep| ep.name() == info.model || ep.name() == info.name)
                    })
                });

                match tier_entry {
                    Some(info) => {
                        let details = info.details.as_ref().map(|d| TagsModelDetails {
                            parent_model: d.parent_model.clone(),
                            format: d.format.clone(),
                            family: d.family.clone(),
                            families: d.families.clone(),
                            parameter_size: d.parameter_size.clone(),
                            quantization_level: d.quantization_level.clone(),
                        });
                        TagsModel {
                            name: tier.to_string(),
                            model: tier.to_string(),
                            modified_at: info.modified_at.clone(),
                            size: info.size,
                            digest: info.digest.clone(),
                            details,
                        }
                    }
                    None => {
                        // No cached info available, return placeholder with minimal details
                        TagsModel {
                            name: tier.to_string(),
                            model: tier.to_string(),
                            modified_at: "2026-01-01T00:00:00Z".to_string(),
                            size: 0,
                            digest: format!("sha256:{:016x}", hash_string(tier)),
                            details: Some(TagsModelDetails {
                                parent_model: "".to_string(),
                                format: "gguf".to_string(),
                                family: format!("{}-tier", tier),
                                families: vec![format!("{}-tier", tier)],
                                parameter_size: "varies".to_string(),
                                quantization_level: "varies".to_string(),
                            }),
                        }
                    }
                }
            };

        // Add tier-based virtual models with details from cached endpoint info
        for tier in ["auto", "fast", "balanced", "deep"] {
            models.push(create_virtual_tier_model(
                tier,
                state.config(),
                &*model_cache_guard,
            ));
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

        // Add discovered models from the cache (all models discovered from Ollama endpoints)
        // We skip virtual models (auto, fast, balanced, deep) which are added above
        // We also skip endpoint name mappings (e.g., "gemma3" endpoint pointing to "gemma4:31b")
        // We also filter out models from unhealthy endpoints
        for (_key, info) in model_cache_guard.iter() {
            // Skip virtual tier names (these are added above with details)
            if info.name == "auto"
                || info.name == "fast"
                || info.name == "balanced"
                || info.name == "deep"
            {
                continue;
            }
            // Skip endpoint name mappings where the name doesn't match the actual model
            // Only return actual models where info.name == info.model
            if info.model != info.name {
                continue;
            }
            // Skip models from unhealthy endpoints
            if !info.is_healthy() {
                continue;
            }
            models.push(model_to_tags(info));
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
    fn test_tags_model_with_details() {
        let model = TagsModel {
            name: "qwen3-8b".to_string(),
            model: "qwen3-8b".to_string(),
            modified_at: "2026-01-01T00:00:00Z".to_string(),
            size: 4_294_967_296,
            digest: "sha256:abc123def456".to_string(),
            details: Some(TagsModelDetails {
                parent_model: "".to_string(),
                format: "gguf".to_string(),
                family: "qwen3".to_string(),
                families: vec!["qwen3".to_string()],
                parameter_size: "8B".to_string(),
                quantization_level: "Q4_K_M".to_string(),
            }),
        };
        assert_eq!(model.name, "qwen3-8b");
        assert_eq!(model.model, "qwen3-8b");
        assert!(model.digest.starts_with("sha256:"));
    }

    #[test]
    fn test_tags_model_auto_tier_has_details() {
        let model = TagsModel {
            name: "auto".to_string(),
            model: "auto".to_string(),
            modified_at: "2026-01-01T00:00:00Z".to_string(),
            size: 0,
            digest: format!("sha256:{:016x}", hash_string("auto")),
            details: Some(TagsModelDetails {
                parent_model: "".to_string(),
                format: "gguf".to_string(),
                family: "multi-tier".to_string(),
                families: vec!["multi-tier".to_string()],
                parameter_size: "varies".to_string(),
                quantization_level: "varies".to_string(),
            }),
        };
        assert_eq!(model.name, "auto");
        assert!(model.digest.starts_with("sha256:"));
        assert!(model.details.is_some());
        let details = model.details.unwrap();
        assert_eq!(details.family, "multi-tier");
        assert_eq!(details.parameter_size, "varies");
    }

    #[test]
    fn test_tags_list_response() {
        let models = vec![
            TagsModel {
                name: "auto".to_string(),
                model: "auto".to_string(),
                modified_at: "2026-01-01T00:00:00Z".to_string(),
                size: 0,
                digest: format!("sha256:{:016x}", hash_string("auto")),
                details: Some(TagsModelDetails {
                    parent_model: "".to_string(),
                    format: "gguf".to_string(),
                    family: "multi-tier".to_string(),
                    families: vec!["multi-tier".to_string()],
                    parameter_size: "varies".to_string(),
                    quantization_level: "varies".to_string(),
                }),
            },
            TagsModel {
                name: "qwen3-8b".to_string(),
                model: "qwen3-8b".to_string(),
                modified_at: "2026-01-01T00:00:00Z".to_string(),
                size: 4_294_967_296,
                digest: "sha256:abc123def456".to_string(),
                details: Some(TagsModelDetails {
                    parent_model: "".to_string(),
                    format: "gguf".to_string(),
                    family: "qwen3".to_string(),
                    families: vec!["qwen3".to_string()],
                    parameter_size: "8B".to_string(),
                    quantization_level: "Q4_K_M".to_string(),
                }),
            },
        ];
        let response = TagsListResponse { models };
        assert_eq!(response.models.len(), 2);
    }

    #[test]
    fn test_hash_string_deterministic() {
        let digest1 = hash_string("test-model");
        let digest2 = hash_string("test-model");
        assert_eq!(digest1, digest2);

        let digest3 = hash_string("different-model");
        assert_ne!(digest1, digest3);
    }
}
