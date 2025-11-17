pub mod depth_anything3;
pub mod depth_pro;

use burn::{
    module::Module,
    prelude::*,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder},
};
use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use std::path::Path;

use depth_anything3::DepthAnything3Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthModelKind {
    DepthPro,
    DepthAnything3,
}

impl DepthModelKind {
    pub fn default_checkpoint(self) -> &'static str {
        match self {
            DepthModelKind::DepthPro => "assets/model/depth_pro.mpk",
            DepthModelKind::DepthAnything3 => "assets/model/da3_metric_large.mpk",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DepthModelKind::DepthPro => "depth-pro",
            DepthModelKind::DepthAnything3 => "depth-anything-3",
        }
    }
}

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
        match kind {
            DepthModelKind::DepthPro => depth_pro::DepthPro::<B>::load(device, checkpoint)
                .map(Self::DepthPro)
                .map_err(|err| format!("Failed to load DepthPro checkpoint: {err}")),
            DepthModelKind::DepthAnything3 => Self::load_depth_anything3(device, checkpoint),
        }
    }

    fn load_depth_anything3(device: &B::Device, checkpoint: &Path) -> Result<Self, String> {
        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let checkpoint_hint = checkpoint
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();

        let mut configs = Vec::from([
            DepthAnything3Config::metric_large(),
            DepthAnything3Config::small(),
        ]);

        if checkpoint_hint.contains("small") {
            configs.swap(0, 1);
        }

        let mut last_err = None;
        for config in configs {
            let config_clone = config.clone();
            let recorder_clone = recorder.clone();
            let attempt = depth_anything3::with_model_load_stack(move || {
                depth_anything3::DepthAnything3::new(device, config_clone)
                    .load_file(checkpoint, &recorder_clone, device)
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

    pub fn kind(&self) -> DepthModelKind {
        match self {
            Self::DepthPro(_) => DepthModelKind::DepthPro,
            Self::DepthAnything3(_) => DepthModelKind::DepthAnything3,
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

    let mut resized = imageops::resize(
        image,
        scaled_width as u32,
        scaled_height as u32,
        FilterType::CatmullRom,
    );

    let crop_x = (scaled_width.saturating_sub(target)) / 2;
    let crop_y = (scaled_height.saturating_sub(target)) / 2;
    let cropped = imageops::crop_imm(
        &mut resized,
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
