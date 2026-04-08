//! Model cache structures for Ollama API compatibility
//!
//! This module provides the structures to cache model information from Ollama endpoints
//! during health checks, which is then used by the /api/tags endpoint to return
//! accurate model metadata to clients.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Model details returned by Ollama's /api/tags endpoint
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

/// Full model information cached from Ollama endpoints
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub digest: String,
    pub size: u64,
    pub details: Option<ModelDetails>,
    /// True if this model was discovered from a healthy endpoint
    /// Skipped in serialization since this is internal tracking only
    #[serde(skip_serializing)]
    pub healthy: bool,
}

/// Creates a new healthy ModelInfo
///
/// Use this factory function when caching models from healthy endpoints.
pub fn new_model_info(
    name: String,
    model: String,
    modified_at: String,
    digest: String,
    size: u64,
    details: Option<ModelDetails>,
) -> ModelInfo {
    ModelInfo {
        name,
        model,
        modified_at,
        digest,
        size,
        details,
        healthy: true,
    }
}

/// Creates a ModelInfo marked as unhealthy (from unhealthy endpoint)
pub fn new_unhealthy_model_info(
    name: String,
    model: String,
    modified_at: String,
    digest: String,
    size: u64,
    details: Option<ModelDetails>,
) -> ModelInfo {
    ModelInfo {
        name,
        model,
        modified_at,
        digest,
        size,
        details,
        healthy: false,
    }
}

/// Cache for storing model information from Ollama endpoints
///
/// This cache is populated during health checks when querying
/// Ollama endpoints for /api/tags. The tags handler uses this
/// cache to return accurate model information to clients.
pub type ModelCache = Arc<RwLock<HashMap<String, ModelInfo>>>;

/// Extension trait for ModelInfo to filter healthy models
pub trait ModelInfoExt {
    fn is_healthy(&self) -> bool;
}

impl ModelInfoExt for ModelInfo {
    fn is_healthy(&self) -> bool {
        self.healthy
    }
}
