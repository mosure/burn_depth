use burn::prelude::*;

use crate::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};

#[derive(Clone, Debug)]
pub struct ViTConfig {
    pub in_chans: usize,
    pub embed_dim: usize,
    pub img_size: usize,
    pub patch_size: usize,
    pub encoder_feature_layer_ids: Vec<usize>,
    pub encoder_feature_dims: Vec<usize>,
}

impl ViTConfig {
    pub fn grid_size(&self) -> usize {
        self.img_size / self.patch_size
    }
}

pub const DINOV2_L16_384: &str = "dinov2l16_384";
pub const DINOV2_L16_128: &str = "dinov2l16_128";

fn vit_config_from_preset(preset: &str) -> Option<ViTConfig> {
    match preset {
        DINOV2_L16_384 => Some(ViTConfig {
            in_chans: 3,
            embed_dim: 1024,
            img_size: 384,
            patch_size: 16,
            encoder_feature_layer_ids: vec![5, 11, 17, 23],
            encoder_feature_dims: vec![256, 512, 1024, 1024],
        }),
        DINOV2_L16_128 => Some(ViTConfig {
            in_chans: 3,
            embed_dim: 1024,
            img_size: 128,
            patch_size: 16,
            encoder_feature_layer_ids: vec![5, 11, 17, 23],
            encoder_feature_dims: vec![256, 512, 1024, 1024],
        }),
        _ => None,
    }
}

pub fn create_vit<B: Backend>(
    device: &B::Device,
    preset: &str,
) -> (DinoVisionTransformer<B>, ViTConfig) {
    let config = vit_config_from_preset(preset)
        .unwrap_or_else(|| panic!("unsupported ViT preset `{preset}`"));

    let vit = match preset {
        DINOV2_L16_384 => {
            DinoVisionTransformerConfig::vitl(Some(config.img_size), Some(config.patch_size))
                .init(device)
        }
        DINOV2_L16_128 => {
            DinoVisionTransformerConfig::vitl(Some(config.img_size), Some(config.patch_size))
                .init(device)
        }
        // Safety: unreachable due to unwrap above.
        _ => unreachable!(),
    };

    (vit, config)
}
