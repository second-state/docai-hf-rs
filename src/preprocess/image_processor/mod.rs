mod common;

use std::fs::File;
use std::io::Read;
use image::ImageBuffer;
use common::*;

pub trait ImageProcessor<'model>: 'model {
    fn process_image(&self, image_path:&str)->Vec<u8>;
    fn image_tensor_type(&self) -> crate::TensorType;
    fn image_tensor_dims(&self) -> &'model [usize];
}

// DeitImageProcessor
#[allow(dead_code)]
pub struct DeitImageProcessor<'model>{
    do_resize : bool,
    do_normalize : bool,
    crop_image_width : f32,
    crop_image_height : f32,
    resize_filter : image::imageops::FilterType,
    image_mean : Vec<f32>,
    image_std : Vec<f32>,
    image_tensor_backend : crate::GraphEncoding,
    image_tensor_type : crate::TensorType,
    image_tensor_dims : &'model [usize],
}

impl<'model> Default for DeitImageProcessor<'model> {
    fn default() -> Self {
        DeitImageProcessor {
            do_resize: true,
            do_normalize: true,
            crop_image_width: 224.0,
            crop_image_height: 224.0,
            resize_filter : image::imageops::FilterType::Triangle,
            image_mean : [0.5,0.5,0.5].to_vec(),
            image_std : [0.5,0.5,0.5].to_vec(),
            image_tensor_backend: crate::GraphEncoding::Pytorch,  
            image_tensor_type: crate::TensorType::F32, 
            image_tensor_dims: &[1,3,224,224], 
            // image_tensor_dims : & 'model[1,3,224,224],   
        }
    }
}

impl<'model> ImageProcessor<'model> for DeitImageProcessor<'model>{
    fn image_tensor_type(&self) -> crate::TensorType {
        self.image_tensor_type
    }

    fn image_tensor_dims(&self) -> &'model [usize] {
        self.image_tensor_dims
    }

    fn process_image(&self, image_path : &str) -> Vec<u8>{
        let mut file_img = File::open(image_path).unwrap();
        let mut img_buf = Vec::new();
        file_img.read_to_end(&mut img_buf).unwrap();
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> = image::load_from_memory(&img_buf).unwrap().to_rgb8();
        let res_img:ImageBuffer<image::Rgb<u8>, Vec<u8>>;
        let flattened_image:Vec<f32>;
        
        if self.do_resize{
            res_img = resize_image(&img, self.crop_image_width, self.crop_image_height, self.resize_filter);
            // resize_image(&img, self.crop_image_width, self.crop_image_height, self.resize_filter);
        }
        else{
            res_img = img;
        }
        if self.do_normalize{
            flattened_image = to_norm_bgr_image(res_img, &self.image_mean , &self.image_std);
        }
        else{
            flattened_image = to_flat_bgr_image(res_img);
        }
        let img_tensor_data = image_to_tensor_data(flattened_image);
        // let (img_tensor_data, img_dims) = (image_to_tensor_data(flattened_image), [1, 3, self.crop_image_width as u32, self.crop_image_height as u32]);
        // let img_tensor = to_tensor(&img_tensor_data, &img_dims, self.to_tensor_type);
        
        return img_tensor_data;
    }
}
