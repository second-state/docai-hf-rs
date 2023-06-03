//!
//! # A Rust library for MediaPipe tasks for WasmEdge WASI-NN
//!
//! ## Introduction
//!


mod error;
// mod util;

// #[macro_use]
// mod model;

/// DocumentAI-rs postprocess api, which define the tasks results and implement tensors results to task results.
/// The module also has utils to make use of results, such as drawing utils.
#[macro_use]
pub mod postprocess;
/// DocumentAI-rs preprocess api, which define the tasks input interface (convert media input to tensors) and implement some builtin pre-process function for types.
pub mod preprocess;
/// DocumentAI-rs tasks api, contain audio, vision and text tasks.
pub mod tasks;


pub use error::Error;
pub use wasi_nn_safe::GraphExecutionTarget as Device;
use wasi_nn_safe::{
    Graph, GraphBuilder, GraphEncoding, GraphExecutionContext, SharedSlice, TensorType,
};
// pub use util::SharedSlice;

#[cfg(doc)]
use tasks::*;
