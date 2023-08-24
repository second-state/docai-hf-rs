use super::*;

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
pub(crate) fn sort_classification_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

#[allow(dead_code)]
pub(crate) struct OutputBuffer{
    pub(crate) data_buffer: Vec<u8>,
    pub(crate) tensor_type: TensorType,
}

macro_rules! output_buffer_mut_slice {
    ( $out:expr) => {
        // match $out.tensor_type {
        //     TensorType::U8 => {
        //         $out.data_buffer.as_mut_slice()
        //     }
            // TensorType::F32 => 
            unsafe {
                core::slice::from_raw_parts_mut(
                    $out.data_buffer.as_mut_slice().as_ptr() as *mut f32,
                    $out.data_buffer.len() >> 2,
                )
            } //,
            // _ => {
            //     todo!("FP16, I32")
            // }
        // }
    };
}

macro_rules! empty_output_buffer {
    ( $x:expr, $elem_size:expr ) => {{
        let bytes_size = tensor_byte_size!($x) * $elem_size;
        OutputBuffer {
            data_buffer: vec![0; bytes_size],
            tensor_type: $x,
        }
    }};
}

macro_rules! tensor_byte_size {
    ($tensor_type:expr) => {
        match $tensor_type {
            crate::TensorType::F32 => 4,
            crate::TensorType::U8 => 1,
            crate::TensorType::I32 => 4,
            crate::TensorType::F16 => 2,
        }
    };
}
