#[macro_use]
mod common;
pub(crate) use common::*;

use super::*;
use crate::{Error, TensorType};

// use super::{InferenceResult, OutputBuffer, DIT_CLASSIFICATION_CLASSES};

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

// ClassificationOutputProcessor
pub struct ClassificationOutputProcessor<'model>{
    pub(crate) max_results: i32,
    pub(crate) labels_asset : [&'model str; 16],
    pub(crate) outputs : Vec<OutputBuffer>,
    // pub(crate) output_tensor_shape: Option<&'a [usize]>,
    // pub(crate) output_buffer_max_size : u32,
}

// impl<'a> Default for DeitImageProcessor<'a> {
//     fn default() -> Self {
//         DeitImageProcessor {
//             do_resize: true,
//             do_normalize: true,
//             crop_image_width: 224.0,
//             crop_image_height: 224.0,
//             resize_filter : image::imageops::FilterType::Triangle,
//             image_mean : [0.5,0.5,0.5].to_vec(),
//             image_std : [0.5,0.5,0.5].to_vec(),
//             image_tensor_backend: crate::GraphEncoding::Pytorch,  
//             image_tensor_type: crate::TensorType::F32, 
//             image_tensor_dims: &[1,3,224,224], 
//             // image_tensor_dims : & 'a[1,3,224,224],   
//         }
//     }
// }

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

    pub(crate) fn process_output(&mut self)->Result<Vec<InferenceResult>,Error>{
        let out = self.outputs.get_mut(0).unwrap();
        // let out_data_slice = self.outputs.get_mut(0).unwrap().data_buffer.as_mut_slice()
        let scores = output_buffer_mut_slice!(out);

        let results = sort_classification_results(scores);
        println!(
            "The given image is predicted to be a \n\n\n   {}.) [{}]({:.4}){}",
            1,
            results[0].0,
            results[0].1,
            self.labels_asset[results[0].0]
        );
        Ok(results)
    }
}
