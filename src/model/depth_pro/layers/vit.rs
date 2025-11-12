use burn::prelude::*;
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};

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

    let mut builder = match preset {
        DINOV2_L16_384 | DINOV2_L16_128 => {
            DinoVisionTransformerConfig::vitl(Some(config.img_size), Some(config.patch_size))
        }
        // Safety: unreachable due to unwrap above.
        _ => unreachable!(),
    };

    builder.block_config.attn.quiet_softmax = false;
    builder.register_token_count = 0;
    builder.use_register_tokens = false;
    builder.normalize_intermediate_tokens = false;

    let vit = builder.init(device);

    (vit, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = crate::InferenceBackend;

    #[test]
    fn dinov2_patch_count_matches_grid() {
        let device = <TestBackend as Backend>::Device::default();
        let (vit, config) = create_vit::<TestBackend>(&device, DINOV2_L16_384);
        let grid = config.grid_size();

        let input = Tensor::<TestBackend, 4>::ones(
            [1, config.in_chans, config.img_size, config.img_size],
            &device,
        );
        let output = vit.forward(input, None);
        let dims: [usize; 3] = output.x_norm_patchtokens.shape().dims();

        assert_eq!(
            dims[1],
            grid * grid,
            "patch tokens ({}) did not match expected grid size ({})",
            dims[1],
            grid * grid
        );
    }
}
