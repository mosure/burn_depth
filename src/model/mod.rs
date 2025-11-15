pub mod depth_anything3;
pub mod depth_pro;

use burn::{
    module::Module,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
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
            DepthModelKind::DepthAnything3 => {
                let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
                let model = depth_anything3::DepthAnything3::new(
                    device,
                    DepthAnything3Config::metric_large(),
                )
                .load_file(checkpoint, &recorder, device)
                .map_err(|err| format!("Failed to load Depth Anything 3 checkpoint: {err}"))?;
                Ok(Self::DepthAnything3(model))
            }
        }
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
}
