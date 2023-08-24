use wasi_nn::Error as WasiNNError;

// /// wasi-nn API error enum
// #[derive(thiserror::Error, Debug)]
// pub enum BackendError {
//     #[error("WASI-NN Backend Error: Caller module passed an invalid argument")]
//     InvalidArgument,
//     #[error("WASI-NN Backend Error: Invalid Encoding")]
//     InvalidEncoding,
//     #[error("WASI-NN Backend Error: Caller module is missing a memory export")]
//     MissingMemory,
//     #[error("WASI-NN Backend Error: Device or resource busy")]
//     Busy,
//     #[error("WASI-NN Backend Error: Runtime Error")]
//     RuntimeError,
//     #[error("Unknown Wasi-NN Backend Error Code `{0}`")]
//     UnknownError(u32),
// }

// impl BackendError {
//     #[inline(always)]
//     pub(crate) fn from(value: u32) -> Self {
//         match value {
//             1 => Self::InvalidArgument,
//             2 => Self::InvalidEncoding,
//             3 => Self::MissingMemory,
//             4 => Self::Busy,
//             5 => Self::RuntimeError,
//             _ => Self::UnknownError(value),
//         }
//     }
// }

// #[derive(thiserror::Error, Debug)]
// pub enum WasiNNError {
//     #[error("IO Error: {0}")]
//     IoError(#[from] std::io::Error),

//     #[error(
//         "Invalid Tensor: Expect data buffer has at least `{expect}` bytes, but it has only `actual` bytes "
//     )]
//     InvalidTensorError { expect: usize, actual: usize },

//     #[error("Backend Error: {0}")]
//     BackendError(#[from] BackendError),
// }

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
