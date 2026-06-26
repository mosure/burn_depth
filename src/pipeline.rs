use burn::prelude::*;
use image::{DynamicImage, imageops::FilterType};

#[cfg(target_arch = "wasm32")]
use crate::loader::resolve_checkpoint_bytes_async;
use crate::{
    inference::{DepthModel, DepthPrediction, rgb_to_input_tensor},
    loader::{DepthLoadConfig, DepthLoadError, DepthLoadEvent, DepthLoadStage, resolve_checkpoint},
    model::AnyDepthModel,
};

#[derive(Clone, Debug, Default)]
pub struct DepthRuntimeConfig {
    pub output_size: Option<(u32, u32)>,
    pub return_gpu_tensors: bool,
}

#[derive(Debug)]
pub enum DepthPipelineError {
    Load(DepthLoadError),
    Backend(String),
    Model(String),
    Image(String),
}

impl std::fmt::Display for DepthPipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(err) => write!(f, "{err}"),
            Self::Backend(err) => write!(f, "{err}"),
            Self::Model(err) => write!(f, "{err}"),
            Self::Image(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for DepthPipelineError {}

impl From<DepthLoadError> for DepthPipelineError {
    fn from(value: DepthLoadError) -> Self {
        Self::Load(value)
    }
}

pub struct DepthPipeline<B: Backend> {
    model: AnyDepthModel<B>,
    device: B::Device,
}

impl<B: Backend> DepthPipeline<B> {
    pub fn load(device: &B::Device, config: DepthLoadConfig) -> Result<Self, DepthPipelineError> {
        Self::load_with_progress(device, config, |_| {})
    }

    pub fn load_with_progress(
        device: &B::Device,
        config: DepthLoadConfig,
        mut progress: impl FnMut(DepthLoadEvent),
    ) -> Result<Self, DepthPipelineError> {
        validate_backend_requirement::<B>(&config)?;

        let checkpoint = resolve_checkpoint(&config, Some(&mut progress))?;
        progress(DepthLoadEvent::new(
            DepthLoadStage::Deserialize,
            format!("loading {}", checkpoint.display()),
        ));
        let model =
            AnyDepthModel::load_with_precision(config.model, device, &checkpoint, config.precision)
                .map_err(DepthPipelineError::Model)?;
        progress(DepthLoadEvent::new(
            DepthLoadStage::ModelReady,
            "model ready",
        ));
        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    pub async fn load_async(
        device: &B::Device,
        config: DepthLoadConfig,
    ) -> Result<Self, DepthPipelineError> {
        Self::load_async_with_progress(device, config, |_| {}).await
    }

    pub async fn load_async_with_progress(
        device: &B::Device,
        config: DepthLoadConfig,
        progress: impl FnMut(DepthLoadEvent),
    ) -> Result<Self, DepthPipelineError> {
        #[cfg(target_arch = "wasm32")]
        {
            let mut progress = progress;
            validate_backend_requirement::<B>(&config)?;
            let artifact = resolve_checkpoint_bytes_async(&config, Some(&mut progress)).await?;
            progress(DepthLoadEvent::new(
                DepthLoadStage::Deserialize,
                format!("loading {}", artifact.name),
            ));
            let model = AnyDepthModel::load_with_precision_from_bytes(
                config.model,
                device,
                &artifact.name,
                artifact.bytes,
                config.precision,
            )
            .map_err(DepthPipelineError::Model)?;
            progress(DepthLoadEvent::new(
                DepthLoadStage::ModelReady,
                "model ready",
            ));
            return Ok(Self {
                model,
                device: device.clone(),
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::load_with_progress(device, config, progress)
        }
    }

    pub fn predict(
        &self,
        image: DynamicImage,
        config: DepthRuntimeConfig,
    ) -> Result<DepthPrediction<B>, DepthPipelineError> {
        let image = if let Some((width, height)) = config.output_size {
            image.resize_exact(width, height, FilterType::Triangle)
        } else {
            image
        };
        let rgb = image.to_rgb8();
        let prepared = self
            .model
            .prepare_input_image(&rgb)
            .map_err(DepthPipelineError::Image)?;
        let input = rgb_to_input_tensor::<B>(
            prepared.rgb.as_raw(),
            prepared.rgb.width() as usize,
            prepared.rgb.height() as usize,
            &self.device,
        )
        .map_err(DepthPipelineError::Image)?;
        let mut prediction = self.model.infer_depth(input);
        prediction.metadata.model = self.model.kind();
        prediction.metadata.output_size = Some((prepared.width as u32, prepared.height as u32));
        prediction.metadata.returned_gpu_tensors = config.return_gpu_tensors;
        Ok(prediction)
    }

    pub fn model(&self) -> &AnyDepthModel<B> {
        &self.model
    }
}

fn backend_name<B: Backend>() -> &'static str {
    std::any::type_name::<B>()
}

fn validate_backend_requirement<B: Backend>(
    config: &DepthLoadConfig,
) -> Result<(), DepthPipelineError> {
    if config.require_gpu
        && !backend_name::<B>().contains("Wgpu")
        && !backend_name::<B>().contains("Cuda")
    {
        return Err(DepthPipelineError::Backend(format!(
            "GPU was required but backend `{}` is not a GPU backend",
            backend_name::<B>()
        )));
    }
    Ok(())
}
