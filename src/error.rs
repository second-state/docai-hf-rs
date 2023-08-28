use wasi_nn::Error as WasiNNError;

/// DocumentAI-rs API error enum.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Wasi-NN Error: {0}")]
    WasiNNError(#[from] WasiNNError),

    #[error("Argument Error: {0}")]
    ArgumentError(String),

    #[error("Model Binary Parse Error: {0}")]
    ModelParseError(String),

    #[error("ZIP File Parse Error: {0}")]
    ZipFileParseError(String),

    #[error("Model Inconsistent Error: {0}")]
    ModelInconsistentError(String),
}
