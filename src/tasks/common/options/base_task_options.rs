// "This file is licensed under Apache 2.0 and it is originally developed by yanghaku for [WasmEdge/mediapipe-rs](https://github.com/WasmEdge/mediapipe-rs)"

// A file to define the basic options used by each 
// task such as string path of model weights 

pub(crate) struct BaseTaskOptions {
    /// The device to run the models.
    pub device: crate::Device,
}

impl Default for BaseTaskOptions {
    fn default() -> Self {
        Self {
            device: crate::Device::CPU,                   // default is CPU because only tested for that  
        }
    }
}

macro_rules! base_task_options_impl {
    ( $TypeName:ident ) => {
        /// Set execution device to run the models.
        #[inline(always)]
        pub fn execution_target(mut self, device: crate::Device) -> Self {
            self.base_task_options.device = device;
            self
        }

        /// Set ```CPU``` device to run the models. (Default device)
        #[inline(always)]
        pub fn cpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::CPU;
            self
        }

        /// Set ```GPU``` device to run the models.
        #[inline(always)]
        pub fn gpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::GPU;
            self
        }

        /// Set ```TPU``` device to run the models.
        #[inline(always)]
        pub fn tpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::TPU;
            self
        }

        /// Use the current build options, read model from file to create a new task instance.
        #[inline(always)]
        pub fn build_from_file(
            self,
            file_path: impl AsRef<std::path::Path>,
        ) -> Result<$TypeName<'model>, crate::Error> {
            self.build_from_buffer(std::fs::read(file_path)?)
        }
    };
}

macro_rules! base_task_options_get_impl {
    () => {
        /// Get the task running device.
        #[inline(always)]
        pub fn device(&self) -> crate::Device {
            self.build_options.base_task_options.device
        }
    };
}

macro_rules! model_base_check_impl {
    ( $model_resource:ident, $expect_input_count:expr, $expect_output_count:expr ) => {{
        let input_tensor_count = $model_resource.input_tensor_count();
        if input_tensor_count != $expect_input_count {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Expect model input tensor count `{}`, but got `{}`",
                $expect_input_count, input_tensor_count
            )));
        }
        let output_tensor_count = $model_resource.output_tensor_count();
        if output_tensor_count != $expect_output_count {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Expect model output tensor count `{}`, but got `{}`",
                $expect_output_count, output_tensor_count
            )));
        }
    }};

    ( $model_resource:ident, $expect_output_count:expr ) => {{
        let output_tensor_count = $model_resource.output_tensor_count();
        if output_tensor_count != $expect_output_count {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Expect model output tensor count `{}`, but got `{}`",
                $expect_output_count, output_tensor_count
            )));
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

macro_rules! tensor_bytes {
    ( $tensor_type:expr, $tensor_shape:ident ) => {{
        let mut b = tensor_byte_size!($tensor_type);
        for s in $tensor_shape {
            b *= s;
        }
        b
    }};
}
