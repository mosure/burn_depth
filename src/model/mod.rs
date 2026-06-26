pub mod depth_anything3;
pub mod depth_pro;

use crate::loader::DepthPrecision;
#[cfg(feature = "bpk")]
use burn::tensor::Bytes;
use burn::{
    module::Module,
    prelude::*,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder},
};
#[cfg(feature = "bpk")]
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use std::path::Path;

use depth_anything3::DepthAnything3Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthModelKind {
    DepthPro,
    DepthAnything3MetricLarge,
    #[deprecated(note = "use DepthAnything3MetricLarge")]
    DepthAnything3,
}

impl DepthModelKind {
    pub fn default_checkpoint(self) -> &'static str {
        match self {
            DepthModelKind::DepthPro => "models/depth-pro/depth_pro.bpk",
            DepthModelKind::DepthAnything3MetricLarge | DepthModelKind::DepthAnything3 => {
                "models/da3/da3_metric_large.bpk"
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DepthModelKind::DepthPro => "depth-pro",
            DepthModelKind::DepthAnything3MetricLarge | DepthModelKind::DepthAnything3 => {
                "depth-anything-3-metric-large"
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum AnyDepthModel<B: Backend> {
    DepthPro(depth_pro::DepthPro<B>),
    DepthAnything3(depth_anything3::DepthAnything3<B>),
}

impl<B: Backend> AnyDepthModel<B> {
    pub fn load(
        kind: DepthModelKind,
        device: &B::Device,
        checkpoint: &Path,
    ) -> Result<Self, String> {
        Self::load_with_precision(kind, device, checkpoint, DepthPrecision::F32)
    }

    pub fn load_with_precision(
        kind: DepthModelKind,
        device: &B::Device,
        checkpoint: &Path,
        precision: DepthPrecision,
    ) -> Result<Self, String> {
        if is_burnpack(checkpoint) {
            return Self::load_burnpack(kind, device, checkpoint, precision);
        }

        match kind {
            DepthModelKind::DepthPro => depth_pro::DepthPro::<B>::load(device, checkpoint)
                .map(Self::DepthPro)
                .map_err(|err| format!("Failed to load DepthPro checkpoint: {err}")),
            DepthModelKind::DepthAnything3MetricLarge | DepthModelKind::DepthAnything3 => {
                Self::load_depth_anything3(device, checkpoint)
            }
        }
    }

    #[cfg(feature = "bpk")]
    pub fn load_with_precision_from_bytes(
        kind: DepthModelKind,
        device: &B::Device,
        artifact_name: &str,
        bytes: Vec<u8>,
        precision: DepthPrecision,
    ) -> Result<Self, String> {
        match kind {
            DepthModelKind::DepthPro => {
                let mut store = burnpack_load_store_from_bytes(bytes, precision);
                Self::load_depth_pro_burnpack_store(device, &mut store)
            }
            DepthModelKind::DepthAnything3MetricLarge | DepthModelKind::DepthAnything3 => {
                Self::load_depth_anything3_burnpack_bytes(device, artifact_name, bytes, precision)
            }
        }
    }

    #[cfg(not(feature = "bpk"))]
    pub fn load_with_precision_from_bytes(
        _kind: DepthModelKind,
        _device: &B::Device,
        artifact_name: &str,
        _bytes: Vec<u8>,
        _precision: DepthPrecision,
    ) -> Result<Self, String> {
        Err(format!(
            "BurnPack checkpoint `{artifact_name}` requires the `bpk` feature"
        ))
    }

    #[cfg(feature = "bpk")]
    fn load_burnpack(
        kind: DepthModelKind,
        device: &B::Device,
        checkpoint: &Path,
        precision: DepthPrecision,
    ) -> Result<Self, String> {
        match kind {
            DepthModelKind::DepthPro => {
                let mut store = burnpack_load_store(checkpoint, precision);
                Self::load_depth_pro_burnpack_store(device, &mut store)
            }
            DepthModelKind::DepthAnything3MetricLarge | DepthModelKind::DepthAnything3 => {
                Self::load_depth_anything3_burnpack(device, checkpoint, precision)
            }
        }
    }

    #[cfg(feature = "bpk")]
    fn load_depth_pro_burnpack_store(
        device: &B::Device,
        store: &mut BurnpackStore,
    ) -> Result<Self, String> {
        let mut model = depth_pro::DepthPro::<B>::new(device, depth_pro::DepthProConfig::default());
        model
            .load_from(store)
            .map_err(|err| format!("Failed to load DepthPro BurnPack checkpoint: {err}"))?;
        Ok(Self::DepthPro(model))
    }

    #[cfg(not(feature = "bpk"))]
    fn load_burnpack(
        _kind: DepthModelKind,
        _device: &B::Device,
        checkpoint: &Path,
        _precision: DepthPrecision,
    ) -> Result<Self, String> {
        Err(format!(
            "BurnPack checkpoint `{}` requires the `bpk` feature",
            checkpoint.display()
        ))
    }

    fn load_depth_anything3(device: &B::Device, checkpoint: &Path) -> Result<Self, String> {
        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let checkpoint_hint = checkpoint
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let configs = depth_anything3_configs_for_hint(&checkpoint_hint);

        let mut last_err = None;
        for config in configs {
            let config_clone = config.clone();
            let recorder_clone = recorder.clone();
            let attempt = depth_anything3::with_model_load_stack(move || {
                depth_anything3::DepthAnything3::new(device, config_clone).load_file(
                    checkpoint,
                    &recorder_clone,
                    device,
                )
            });
            match attempt {
                Ok(model) => return Ok(Self::DepthAnything3(model)),
                Err(err) => last_err = Some(err),
            }
        }

        Err(format!(
            "Failed to load Depth Anything 3 checkpoint `{}`: {}",
            checkpoint.display(),
            last_err
                .map(|err| err.to_string())
                .unwrap_or_else(|| "unknown error".to_string())
        ))
    }

    #[cfg(feature = "bpk")]
    fn load_depth_anything3_burnpack(
        device: &B::Device,
        checkpoint: &Path,
        precision: DepthPrecision,
    ) -> Result<Self, String> {
        let artifact_name = checkpoint
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.bpk");
        let configs = depth_anything3_configs_for_hint(artifact_name);

        let mut last_err = None;
        for config in configs {
            let mut store = burnpack_load_store(checkpoint, precision);
            let mut model = depth_anything3::DepthAnything3::new(device, config);
            let attempt = depth_anything3::with_model_load_stack(|| {
                model.load_from(&mut store).map(|_| model)
            });
            match attempt {
                Ok(model) => return Ok(Self::DepthAnything3(model)),
                Err(err) => last_err = Some(err),
            }
        }

        Err(format!(
            "Failed to load Depth Anything 3 BurnPack checkpoint `{}`: {}",
            artifact_name,
            last_err
                .map(|err| err.to_string())
                .unwrap_or_else(|| "unknown error".to_string())
        ))
    }

    #[cfg(feature = "bpk")]
    fn load_depth_anything3_burnpack_bytes(
        device: &B::Device,
        artifact_name: &str,
        bytes: Vec<u8>,
        precision: DepthPrecision,
    ) -> Result<Self, String> {
        let configs = depth_anything3_configs_for_hint(artifact_name);

        let mut last_err = None;
        for config in configs {
            let mut store = burnpack_load_store_from_bytes(bytes.clone(), precision);
            let mut model = depth_anything3::DepthAnything3::new(device, config);
            let attempt = depth_anything3::with_model_load_stack(|| {
                model.load_from(&mut store).map(|_| model)
            });
            match attempt {
                Ok(model) => return Ok(Self::DepthAnything3(model)),
                Err(err) => last_err = Some(err),
            }
        }

        Err(format!(
            "Failed to load Depth Anything 3 BurnPack checkpoint `{}`: {}",
            artifact_name,
            last_err
                .map(|err| err.to_string())
                .unwrap_or_else(|| "unknown error".to_string())
        ))
    }

    pub fn kind(&self) -> DepthModelKind {
        match self {
            Self::DepthPro(_) => DepthModelKind::DepthPro,
            Self::DepthAnything3(_) => DepthModelKind::DepthAnything3MetricLarge,
        }
    }

    pub fn as_depth_pro(&self) -> Option<&depth_pro::DepthPro<B>> {
        if let Self::DepthPro(model) = self {
            Some(model)
        } else {
            None
        }
    }

    pub fn as_depth_anything3(&self) -> Option<&depth_anything3::DepthAnything3<B>> {
        if let Self::DepthAnything3(model) = self {
            Some(model)
        } else {
            None
        }
    }

    pub fn preferred_input_resolution(&self) -> Option<usize> {
        match self {
            Self::DepthPro(_) => None,
            Self::DepthAnything3(model) => Some(model.img_size()),
        }
    }

    pub fn prepare_input_image(&self, image: &RgbImage) -> Result<PreparedModelImage, String> {
        match self {
            Self::DepthPro(_) => Ok(PreparedModelImage {
                width: image.width() as usize,
                height: image.height() as usize,
                rgb: image.clone(),
                crop: None,
            }),
            Self::DepthAnything3(model) => prepare_depth_anything3_image(image, model.img_size()),
        }
    }
}

fn is_burnpack(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("bpk"))
}

fn depth_anything3_configs_for_hint(checkpoint_hint: &str) -> Vec<DepthAnything3Config> {
    let mut configs = Vec::from([
        DepthAnything3Config::metric_large(),
        DepthAnything3Config::small(),
    ]);

    if checkpoint_hint.to_ascii_lowercase().contains("small") {
        configs.swap(0, 1);
    }

    configs
}

#[cfg(feature = "bpk")]
fn burnpack_load_store(checkpoint: &Path, precision: DepthPrecision) -> BurnpackStore {
    let store = BurnpackStore::from_file(checkpoint).auto_extension(false);
    if matches!(precision, DepthPrecision::F16) {
        store.with_from_adapter(HalfPrecisionAdapter::new())
    } else {
        store
    }
}

#[cfg(feature = "bpk")]
fn burnpack_load_store_from_bytes(bytes: Vec<u8>, precision: DepthPrecision) -> BurnpackStore {
    let store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
    if matches!(precision, DepthPrecision::F16) {
        store.with_from_adapter(HalfPrecisionAdapter::new())
    } else {
        store
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageCropRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Debug)]
pub struct PreparedModelImage {
    pub width: usize,
    pub height: usize,
    pub rgb: RgbImage,
    pub crop: Option<ImageCropRegion>,
}

pub fn prepare_depth_anything3_image(
    image: &RgbImage,
    target: usize,
) -> Result<PreparedModelImage, String> {
    if target == 0 {
        return Err("depth_anything3 requires a non-zero target resolution".to_string());
    }
    let (orig_width, orig_height) = (image.width() as usize, image.height() as usize);
    if orig_width == target && orig_height == target {
        return Ok(PreparedModelImage {
            width: target,
            height: target,
            rgb: image.clone(),
            crop: None,
        });
    }

    let shortest = orig_width.min(orig_height).max(1) as f32;
    let scale = target as f32 / shortest;
    let mut scaled_width = ((orig_width as f32) * scale).round() as usize;
    let mut scaled_height = ((orig_height as f32) * scale).round() as usize;
    scaled_width = scaled_width.max(target);
    scaled_height = scaled_height.max(target);

    let resized = imageops::resize(
        image,
        scaled_width as u32,
        scaled_height as u32,
        FilterType::CatmullRom,
    );

    let crop_x = (scaled_width.saturating_sub(target)) / 2;
    let crop_y = (scaled_height.saturating_sub(target)) / 2;
    let cropped = imageops::crop_imm(
        &resized,
        crop_x as u32,
        crop_y as u32,
        target as u32,
        target as u32,
    )
    .to_image();

    Ok(PreparedModelImage {
        width: target,
        height: target,
        rgb: cropped,
        crop: None,
    })
}
