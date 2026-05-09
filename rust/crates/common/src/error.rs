use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("api error: {0}")]
    Api(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("signing error: {0}")]
    Signing(String),

    #[error("invalid order: {0}")]
    InvalidOrder(String),

    #[error("insufficient margin: required {required}, available {available}")]
    InsufficientMargin { required: String, available: String },

    #[error("risk check failed: {0}")]
    RiskRejected(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
