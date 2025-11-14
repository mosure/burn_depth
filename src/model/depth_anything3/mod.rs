use burn::{
    module::{Ignored, Module},
    prelude::*,
};
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};

mod dpt;
mod interpolate;

pub use dpt::{DepthAnything3Head, DepthAnything3HeadConfig, HeadActivation};

#[derive(Config, Debug)]
pub struct DepthAnything3Config {
    pub image_size: usize,
    pub patch_size: usize,
    pub hook_block_ids: Vec<usize>,
    pub head: DepthAnything3HeadConfig,

    #[config(default = "None")]
    pub checkpoint_uri: Option<String>,
}

impl Default for DepthAnything3Config {
    fn default() -> Self {
        Self {
            image_size: 518,
            patch_size: 14,
            hook_block_ids: vec![4, 11, 17, 23],
            head: DepthAnything3HeadConfig::metric_large(),
            checkpoint_uri: Some("assets/model/da3_metric_large.mpk".to_string()),
        }
    }
}

impl DepthAnything3Config {
    pub fn metric_large() -> Self {
        Self::default()
    }
}

#[derive(Module, Debug)]
struct Backbone<B: Backend> {
    pretrained: DinoVisionTransformer<B>,
}

impl<B: Backend> Backbone<B> {
    fn new(device: &B::Device, image_size: usize, patch_size: usize) -> Self {
        let mut config = DinoVisionTransformerConfig::vitl(Some(image_size), Some(patch_size));
        config.register_token_count = 0;
        config.use_register_tokens = false;
        config.block_config.attn.quiet_softmax = false;
        Self {
            pretrained: config.init(device),
        }
    }

    fn forward_with_hooks(
        &self,
        input: Tensor<B, 4>,
        hook_blocks: &[usize],
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        let (output, hooks) = self
            .pretrained
            .forward_with_intermediate_tokens(input, hook_blocks);
        (output.x_norm_patchtokens, hooks)
    }
}

#[derive(Module, Debug)]
pub struct DepthAnything3<B: Backend> {
    backbone: Backbone<B>,
    head: DepthAnything3Head<B>,
    patch_size: Ignored<usize>,
    hook_block_ids: Ignored<Vec<usize>>,
    patch_token_start: Ignored<usize>,
}

pub struct DepthAnything3Inference<B: Backend> {
    pub depth: Tensor<B, 3>,
}

impl<B: Backend> DepthAnything3<B> {
    pub fn new(device: &B::Device, config: DepthAnything3Config) -> Self {
        let backbone = Backbone::new(device, config.image_size, config.patch_size);
        let head = DepthAnything3Head::new(device, config.head.clone());
        Self {
            backbone,
            head,
            patch_size: Ignored(config.patch_size),
            hook_block_ids: Ignored(config.hook_block_ids),
            patch_token_start: Ignored(1),
        }
    }

    pub fn infer(&self, input: Tensor<B, 4>) -> DepthAnything3Inference<B> {
        let dims = input.shape().dims::<4>();
        let height = dims[2];
        let width = dims[3];
        assert_eq!(
            height % self.patch_size.0,
            0,
            "Input height {height} must be divisible by patch size {}",
            self.patch_size.0
        );
        assert_eq!(
            width % self.patch_size.0,
            0,
            "Input width {width} must be divisible by patch size {}",
            self.patch_size.0
        );

        let (_, hooks) = self
            .backbone
            .forward_with_hooks(input, &self.hook_block_ids.0);
        assert!(
            hooks.len() >= self.hook_block_ids.0.len(),
            "Backbone returned fewer hooks ({}) than requested ({})",
            hooks.len(),
            self.hook_block_ids.0.len()
        );
        let depth = self.head.forward(
            &hooks,
            height,
            width,
            self.patch_token_start.0,
            self.patch_size.0,
        );
        DepthAnything3Inference { depth }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceBackend;

    #[test]
    fn depth_anything3_emits_depth_tensor() {
        let device = <InferenceBackend as Backend>::Device::default();
        let config = DepthAnything3Config::metric_large();
        let model = DepthAnything3::<InferenceBackend>::new(&device, config);
        let input = Tensor::<InferenceBackend, 4>::zeros([1, 3, 518, 518], &device);
        let output = model.infer(input);
        assert_eq!(output.depth.shape().dims(), [1, 518, 518]);
    }
}
