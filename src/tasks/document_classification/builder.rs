// "This file refers [WasmEdge/mediapipe-rs](https://github.com/WasmEdge/mediapipe-rs) licensed under Apache 2.0 and originally developed by yanghaku for WasmEdge"

use super::DocumentClassifier;
use crate::tasks::common::BaseTaskOptions;

/// Configure the build options of a new **Document Classification** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct DocumentClassifierBuilder<'model> {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) graph_lifetime: std::marker::PhantomData<&'model ()>,
}

impl<'model> Default for DocumentClassifierBuilder<'model> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            graph_lifetime: std::marker::PhantomData,
        }
    }
}

impl<'model> DocumentClassifierBuilder<'model> {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            graph_lifetime: std::marker::PhantomData,
        }
    }

    base_task_options_impl!(DocumentClassifier);


    /// Use the build options and the buffer as the model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(
        self,
        buffer: impl AsRef<[u8]>,
    ) -> Result<DocumentClassifier<'model>, crate::Error> {
        let buf = buffer.as_ref();

        // // change the lifetime to 'static, because the buf will move to graph and will not be released.
        // let model_resource_ref = crate::model::parse_model(buf.as_ref())?;
        // let model_resource = unsafe {
        //     std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        // };

        // // check model
        // model_base_check_impl!(model_resource, 1, 1);
        // model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let input_tensor_type = crate::TensorType::F32;

        let graph = crate::GraphBuilder::new(
            crate::GraphEncoding::Pytorch,             // currently only tested for Pytorch models from HuggingFace
            self.base_task_options.device,
        )
        .build_from_bytes([buf])?;

        return Ok(DocumentClassifier {
            build_options: self,
            graph,
            input_tensor_type,
        });
    }
}

#[cfg(test)]
mod test {
    use crate::tasks::DocumentClassifierBuilder;

    #[test]
    fn test_builder_check() {
        assert!(DocumentClassifierBuilder::new().build_from_buffer([]).is_err());
        // assert!(DocumentClassifierBuilder::new().build_from_file("").is_err());
    }
}
