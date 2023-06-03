use image::{io::Reader as Reader, DynamicImage, GenericImageView, ImageBuffer};

pub(crate) fn resize_image(img: &ImageBuffer<image::Rgb<u8>, Vec<u8>>, image_width:f32, image_height:f32, resize_filter:image::imageops::FilterType)->ImageBuffer<image::Rgb<u8>, Vec<u8>> { //-> ImageBuffer{
    let resized =
        image::imageops::resize(img, image_width as u32, image_height as u32, resize_filter); // is technically linear but let's check
    // return image::DynamicImage::ImageRgba8(resized);
    return resized;
}

pub(crate) fn to_norm_bgr_image(res_dyn_img: ImageBuffer<image::Rgb<u8>, Vec<u8>>, image_mean: &Vec<f32>, image_std: &Vec<f32>)->Vec<f32>{
    let mut flat_img: Vec<f32> = Vec::new();

    // normalizing wrt mean and std deviation
    for rgb in res_dyn_img.pixels() {
        flat_img.push(((rgb[0] as f32 / 255.) - image_mean[0]) / image_std[0]);
        flat_img.push(((rgb[1] as f32 / 255.) - image_mean[1]) / image_std[1]);
        flat_img.push(((rgb[2] as f32 / 255.) - image_mean[2]) / image_std[2]);
    }
    // // for layoutlmv2
    // // let img_tensor = to_tensor(flat_img, &[1, 3, 224, 224], wasi_nn::TENSOR_TYPE_F32);
    return flat_img;
}

pub(crate) fn to_flat_bgr_image(res_dyn_img: ImageBuffer<image::Rgb<u8>, Vec<u8>>)->Vec<f32>{
    let mut flat_img: Vec<f32> = Vec::new();

    for rgb in res_dyn_img.pixels() {
        flat_img.push(rgb[0] as f32 / 255.);
        flat_img.push(rgb[1] as f32 / 255.);
        flat_img.push(rgb[2] as f32 / 255.);
    }
    return flat_img;
}

// fn to_tensor<'a>(tensor_data:&'a Vec<u8>, tensor_dims:&'a [u32], tensor_type: crate::TensorType)->crate::tensor<'a>{
    
//     let tensor = crate::Tensor {
//         dimensions: tensor_dims,
//         type_: tensor_type,
//         data: tensor_data,
//     };

//     return tensor;
// }

pub(crate) fn image_to_tensor_data(flat_img:Vec<f32>)->Vec<u8>{
    let bytes_required = flat_img.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    for c in 0..3 {
        for i in 0..(flat_img.len() / 3) {
            // Read the number as a f32 and break it into u8 bytes
            let u8_f32: f32 = flat_img[i * 3 + c] as f32;
            let u8_bytes = u8_f32.to_ne_bytes();

            for j in 0..4 {
                u8_f32_arr[((flat_img.len() / 3 * c + i) * 4) + j] = u8_bytes[j];
            }
        }
    }
    return u8_f32_arr;
}
