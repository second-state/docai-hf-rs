mod image_processor;

pub use image_processor::*;

#[allow(dead_code)]
pub struct InputProcessingOptions<'model> {
    pub(crate) image_processor: Box<dyn ImageProcessor<'model>>,
    pub(crate) tokenizer: Option<u32>,
}
