use super::DocumentClassifier;
use crate::tasks::common::{BaseTaskOptions};

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

    base_task_options_impl!();


    /// Use the build options to create a new task instance.
    #[inline]
    pub fn finalize(mut self) -> Result<DocumentClassifier<'model>, crate::Error> {
        let buf = base_task_options_check_and_get_buf!(self);

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
            self.base_task_options.execution_target,
        )
        .build_from_shared_slices([buf])?;

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
        assert!(DocumentClassifierBuilder::new().finalize().is_err());
        assert!(DocumentClassifierBuilder::new()
            .model_asset_buffer("".into())
            .model_asset_path("")
            .finalize()
            .is_err());
        assert!(DocumentClassifierBuilder::new()
            .model_asset_path("")
            .max_results(0)
            .finalize()
            .is_err());
    }
}
