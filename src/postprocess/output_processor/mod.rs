#[macro_use]
mod common;
pub(crate) use common::*;

mod classification_output_processor;
pub use classification_output_processor::*;

use crate::{Error, TensorType, postprocess::ClassificationResult, postprocess::InferenceResult};

//ToDo : Add below options to OutputProcessor 

// pub trait OutputProcessor<'a>: 'a {
//     fn process_output(&'a mut self)-> Result<Vec<InferenceResult>,Error>;
//     fn output_buffer(&mut self, index: usize) -> &mut [u8];
//     fn add_classification_options(
//         &mut self,
//         buffer_tensor_type: TensorType,
//         buffer_shape: &[usize],
//     );
//     // fn image_tensor_type(&self) -> crate::TensorType;
//     // fn image_tensor_dims(&self) -> &'a [usize];
// }
