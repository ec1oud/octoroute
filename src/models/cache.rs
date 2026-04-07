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
}

/// Cache for storing model information from Ollama endpoints
///
/// This cache is populated during health checks when querying
/// Ollama endpoints for /api/tags. The tags handler uses this
/// cache to return accurate model information to clients.
pub type ModelCache = Arc<RwLock<HashMap<String, ModelInfo>>>;
