pub struct InferenceResult(usize, f32);

mod label_assets;
pub use label_assets::*;

#[macro_use]
mod output_processor;
pub use output_processor::*;

mod containers;
pub use containers::*;
