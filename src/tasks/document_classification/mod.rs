mod builder;
use std::process::Output;

pub use builder::DocumentClassifierBuilder;

use crate::preprocess::{InputProcessingOptions, ImageProcessor, DeitImageProcessor};
use crate::postprocess::{OutputBuffer ,InferenceResult, ClassificationOutputProcessor, DIT_CLASSIFICATION_CLASSES};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on images and video frames.
pub struct DocumentClassifier<'model> {
    build_options: DocumentClassifierBuilder<'model>,
    graph: Graph,
    input_tensor_type: TensorType,
}

impl<'model> DocumentClassifier<'model> {
    base_task_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<DocumentClassifierSession, Error> {
        let input_processing_options =  InputProcessingOptions {
                                                                image_processor: Box::new(DeitImageProcessor::default()),
                                                                tokenizer: None,
                                                            };  
        let image_tensor_shape = input_processing_options.image_processor.image_tensor_dims();
        let image_tensor_type = input_processing_options.image_processor.image_tensor_type();

        // let mut output_tensor_buf = vec![0f32; 16];
        let mut output_processor = ClassificationOutputProcessor{
                                                                                    max_results : 1,
                                                                                    labels_asset:DIT_CLASSIFICATION_CLASSES,
                                                                                    outputs:Vec::new(),
                                                                                    // output_buffer_max_size : (output_tensor_buf.len() * 4).try_into().unwrap(),     
                                                                                };
        // The model gives an output of 16 float numbers in one single tensor
        // Thus the buffer in index 0 should have length = bytes to store 16 floating point numbers
        output_processor.add_classification_options(TensorType::F32, &[16]);

        // let output_processing_options = OutputProcessingOptions {
        //                                                          output_processor : classification_output_processor,     
        //                                                        };

        let execution_ctx = self.graph.init_execution_context()?;


        Ok(DocumentClassifierSession {
            execution_ctx,
            output_processor, //output_processing_options,
            input_processing_options,
            image_tensor_shape,
            image_tensor_buf: vec![0; tensor_bytes!(self.input_tensor_type, image_tensor_shape)],
            image_tensor_type,
            // output_buffer_vec : output_processing_options.output_processor.outputs,
            // output_buffer_max_size:classification_output_processor.output_buffer_max_size,
        })
    }

    /// Classify one image using a new session.
    #[inline(always)]
    pub fn classify(&self, input: &str) -> Result<Vec<InferenceResult>, Error> {
        // let x = self.new_session()?;
        // x.classify(input)
        self.new_session()?.classify(input)
    }
}

pub struct DocumentClassifierSession<'model>{
    execution_ctx: GraphExecutionContext<'model>,
    output_processor: ClassificationOutputProcessor<'model>,    //output_processing_options: OutputProcessingOptions<'model>,

    // only one input(image) and one output(for classification)
    input_processing_options: InputProcessingOptions<'model>,
    image_tensor_shape: &'model [usize],
    image_tensor_buf: Vec<u8>,
    image_tensor_type: TensorType,
    // output_buffer_vec : Vec<OutputBuffer>,
    // output_buffer_max_size : u32,
}

impl<'model> DocumentClassifierSession<'model> {
    #[inline(always)]
    fn compute(&mut self)-> Result<Vec<InferenceResult>, Error> {
        self.execution_ctx.set_input(
            0,
            self.image_tensor_type,
            self.image_tensor_shape,
            self.image_tensor_buf.as_ref(),
        )?;

        self.execution_ctx.compute()?;

        // only one output
        let output_buffer = self.output_processor.output_buffer(0);
        let output_size = self.execution_ctx.get_output(0, output_buffer)?;
        if output_size != output_buffer.len() {
            return Err(Error::ModelInconsistentError(format!(
                "Model output bytes size is `{}`, but got `{}`",
                output_buffer.len(),
                output_size
            )));
        }
        // let scores = output_buffer_mut_slice!(output_buffer);

        // let classification_results = self.output_processor.process_output();
        // classification_results
        self.output_processor.process_output()
    }

    /// Classify one image, reuse this session data to speedup.
    #[inline(always)]
    pub fn classify(&mut self, input: &str)-> Result<Vec<InferenceResult>, Error> {
        self.image_tensor_buf = self.input_processing_options.image_processor.process_image(input);
        self.compute()
    }

}
