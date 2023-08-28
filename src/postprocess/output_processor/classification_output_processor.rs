// "This file refers [WasmEdge/mediapipe-rs](https://github.com/WasmEdge/mediapipe-rs) licensed under Apache 2.0 and originally developed by yanghaku for WasmEdge"

use super::*;

// ClassificationOutputProcessor

#[allow(dead_code)]
pub struct ClassificationOutputProcessor<'model>{
    pub(crate) max_results: i32,
    pub(crate) labels_asset : [&'model str; 16],
    pub(crate) outputs : Vec<OutputBuffer>,
    // pub(crate) output_tensor_shape: Option<&'a [usize]>,
    // pub(crate) output_buffer_max_size : u32,
}

impl<'a> ClassificationOutputProcessor<'a>{
    // fn image_tensor_type(&self) -> crate::TensorType {
    //     self.image_tensor_type
    // }

    // fn image_tensor_dims(&self) -> &'a [usize] {
    //     self.image_tensor_dims
    // }

    pub(crate) fn add_classification_options(
        &mut self,
        buffer_tensor_type: TensorType,
        buffer_shape: &[usize],
    ) {
        let elem_size = buffer_shape.iter().fold(1, |a, b| a * b);
        self.outputs
            .push(empty_output_buffer!(buffer_tensor_type, elem_size));
    }

    /// index must be valid. or panic!
    #[inline(always)]
    pub(crate) fn output_buffer(&mut self, index: usize) -> &mut [u8] {
        self.outputs
            .get_mut(index)
            .unwrap()
            .data_buffer
            .as_mut_slice()
    }

    pub(crate) fn process_output(&mut self)->Result<Vec<ClassificationResult>,Error>{
        
        let out = self.outputs.get_mut(0).unwrap();
        // let out_data_slice = self.outputs.get_mut(0).unwrap().data_buffer.as_mut_slice()
        let scores = output_buffer_mut_slice!(out);

        let inference_results = sort_classification_results(scores);

        // ToDo : Add another struct that can display all Classification Results according to options given to it directly. 
        let results: Vec<ClassificationResult> = inference_results
        .iter()
        .map(|result| ClassificationResult {
            category_name: self.labels_asset[result.0].to_string(),
            score: result.1,
            index: result.0,
        })
        .collect();

        Ok(results)
    }
}
