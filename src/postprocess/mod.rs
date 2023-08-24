// pub mod inference_result;


// pub(crate) struct InferenceResult(pub usize, pub f32); //(usize, f32); 
// pub(crate) use InferenceResult;
// pub(crate) use memory_text_file::MemoryTextFile;


// JUST FOR NOW
// pub(crate) struct InferenceResult<'model>(usize, f32, std::marker::PhantomData<&'model ()>,);
pub struct InferenceResult(usize, f32);

mod label_assets;
pub use label_assets::*;

#[macro_use]
mod output_processor;
pub use output_processor::*;

mod containers;
pub use containers::*;

// pub struct OutputProcessingOptions<'model> {
//     pub(crate) output_processor : Box<dyn OutputProcessor<'model>>,
// }

// macro_rules! realloc_output_buffer {
//     ( $self:expr, $new_size:expr ) => {
//         let new_size = $new_size;
//         if let Some(ref mut t) = $self.quantization_parameters {
//             if t.1.len() < new_size {
//                 t.1.resize(new_size, 0f32);
//             }
//         }
//         let s = tensor_byte_size!($self.tensor_type) * new_size;
//         if $self.data_buffer.len() < s {
//             $self.data_buffer.resize(s, 0);
//         }
//     };
// }
