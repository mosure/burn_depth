use burn::{
    module::{Ignored, Module},
    prelude::*,
};
use burn_dino::model::dino::{
    DinoIntermediate, DinoVisionTransformer, DinoVisionTransformerConfig,
};
use std::cell::RefCell;

use camera::{CameraDecoder, CameraEncoder, CameraPrediction};
mod camera;
mod dpt;
mod interpolate;

pub use camera::{CameraDecoderConfig, CameraEncoderConfig};
pub use dpt::{
    DepthAnything3Head, DepthAnything3HeadConfig, DualDepthAnything3Head, DualHeadOutput,
    HeadActivation, PosEmbedCache,
};

mod stack_guard {
    #[cfg(not(target_arch = "wasm32"))]
    use stacker::maybe_grow;

    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_model_load_stack<R>(f: impl FnOnce() -> R) -> R {
        // Windows threads default to a 1MB stack which isn't enough for DepthAnything3's
        // checkpoint load graph on the WGPU backend. Borrow a larger stack temporarily
        // so Module::load_record can recurse safely.
        const STACK_SIZE: usize = 32 * 1024 * 1024;
        const RED_ZONE: usize = 2 * 1024 * 1024;
        maybe_grow(STACK_SIZE, RED_ZONE, f)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn with_model_load_stack<R>(f: impl FnOnce() -> R) -> R {
        f()
    }
}

#[derive(Debug)]
pub struct CachedDepthAnything3<B: Backend> {
    model: DepthAnything3<B>,
    cache: RefCell<PosEmbedCache<B>>,
}

impl<B: Backend> CachedDepthAnything3<B> {
    pub fn new(model: DepthAnything3<B>) -> Self {
        Self {
            model,
            cache: RefCell::new(PosEmbedCache::new()),
        }
    }

    pub fn inner(&self) -> &DepthAnything3<B> {
        &self.model
    }

    pub fn inner_mut(&mut self) -> &mut DepthAnything3<B> {
        &mut self.model
    }

    pub fn into_inner(self) -> DepthAnything3<B> {
        self.model
    }

    fn with_cache<R>(&self, f: impl FnOnce(&mut PosEmbedCache<B>) -> R) -> R {
        let mut cache = self.cache.borrow_mut();
        f(&mut cache)
    }

    pub fn infer(&self, input: Tensor<B, 4>) -> DepthAnything3Inference<B> {
        self.with_cache(|cache| self.model.infer_with_cache(input, cache))
    }

    pub fn infer_with_camera(
        &self,
        input: Tensor<B, 4>,
        extrinsics: Tensor<B, 4>,
        intrinsics: Tensor<B, 4>,
    ) -> DepthAnything3Inference<B> {
        self.with_cache(|cache| {
            self.model
                .infer_with_camera_cache(input, extrinsics, intrinsics, cache)
        })
    }

    pub fn infer_with_trace(
        &self,
        input: Tensor<B, 4>,
    ) -> (DepthAnything3Inference<B>, DepthTrace<B>) {
        self.with_cache(|cache| self.model.infer_with_trace_cache(input, cache))
    }

    pub fn infer_raw(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.with_cache(|cache| self.model.infer_raw_with_cache(input, cache))
    }

    pub fn infer_from_tokens(
        &self,
        patches: &[Tensor<B, 3>],
        height: usize,
        width: usize,
    ) -> (
        DepthAnything3Inference<B>,
        Option<Vec<Tensor<B, 4>>>,
        Option<Tensor<B, 4>>,
        Option<Tensor<B, 4>>,
    ) {
        self.with_cache(|cache| {
            self.model
                .infer_from_tokens_with_cache(patches, height, width, cache)
        })
    }

    pub fn cache_mut(&self) -> std::cell::RefMut<'_, PosEmbedCache<B>> {
        self.cache.borrow_mut()
    }
}

pub use stack_guard::with_model_load_stack;

#[derive(Config, Debug)]
pub struct DepthAnything3Config {
    pub image_size: usize,
    pub patch_size: usize,
    pub hook_block_ids: Vec<usize>,
    pub head: DepthAnything3HeadConfig,
    #[config(default = "None")]
    pub camera_encoder: Option<CameraEncoderConfig>,
    #[config(default = "None")]
    pub camera_decoder: Option<CameraDecoderConfig>,

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
            camera_encoder: None,
            camera_decoder: None,
            checkpoint_uri: Some("assets/model/da3_metric_large.mpk".to_string()),
        }
    }
}

impl DepthAnything3Config {
    pub fn metric_large() -> Self {
        Self::default()
    }

    pub fn small() -> Self {
        Self {
            image_size: 518,
            patch_size: 14,
            hook_block_ids: vec![5, 7, 9, 11],
            head: DepthAnything3HeadConfig::small(),
            camera_encoder: Some(CameraEncoderConfig {
                dim_out: 384,
                ..CameraEncoderConfig::default()
            }),
            camera_decoder: Some(CameraDecoderConfig { dim_in: 768 }),
            checkpoint_uri: Some("assets/model/da3_small.mpk".to_string()),
        }
    }
}

#[derive(Module, Debug)]
struct Backbone<B: Backend> {
    pretrained: DinoVisionTransformer<B>,
}

impl<B: Backend> Backbone<B> {
    fn new(device: &B::Device, config: &DepthAnything3Config) -> Self {
        let mut vit_config = if config.head.dim_in >= 1024 {
            DinoVisionTransformerConfig::vitl(Some(config.image_size), Some(config.patch_size))
        } else {
            DinoVisionTransformerConfig::vits(Some(config.image_size), Some(config.patch_size))
        };
        vit_config.register_token_count = 0;
        vit_config.use_register_tokens = false;
        vit_config.use_mask_token = false;
        vit_config.block_config.attn.quiet_softmax = false;
        if config.head.dual_head {
            vit_config.alt_block_start = Some(4);
            vit_config.qk_norm_block_start = Some(4);
            vit_config.rope_block_start = Some(4);
            vit_config.cat_token = true;
            vit_config.use_camera_tokens = true;
        }
        Self {
            pretrained: vit_config.init(device),
        }
    }

    fn forward_with_hooks(
        &self,
        input: Tensor<B, 4>,
        hook_blocks: &[usize],
        camera_token: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Vec<DinoIntermediate<B>>) {
        let (output, hooks, _) = self.pretrained.forward_with_intermediate_tokens_ext(
            input,
            hook_blocks,
            &[],
            camera_token,
        );
        (output.x_norm_patchtokens, hooks)
    }
}

#[derive(Module, Debug)]
pub struct DepthAnything3<B: Backend> {
    backbone: Backbone<B>,
    head_mono: Option<DepthAnything3Head<B>>,
    head_dual: Option<DualDepthAnything3Head<B>>,
    camera_encoder: Option<CameraEncoder<B>>,
    camera_decoder: Option<CameraDecoder<B>>,
    img_size: Ignored<usize>,
    patch_size: Ignored<usize>,
    hook_block_ids: Ignored<Vec<usize>>,
    patch_token_start: Ignored<usize>,
}

pub struct DepthAnything3Inference<B: Backend> {
    pub depth: Tensor<B, 3>,
    pub depth_confidence: Option<Tensor<B, 3>>,
    pub aux: Option<Tensor<B, 4>>,
    pub aux_confidence: Option<Tensor<B, 3>>,
    pub pose_encoding: Option<Tensor<B, 3>>,
    pub extrinsics: Option<Tensor<B, 4>>,
    pub intrinsics: Option<Tensor<B, 4>>,
}

pub struct DepthTrace<B: Backend> {
    pub backbone_tokens: Vec<Tensor<B, 3>>,
    pub aux_stage_necks: Option<Vec<Tensor<B, 4>>>,
    pub aux_logits: Option<Tensor<B, 4>>,
    pub aux_head_input: Option<Tensor<B, 4>>,
}

enum HeadForwardOutput<B: Backend> {
    Mono(Tensor<B, 4>),
    Dual(DualHeadOutput<B>),
}

impl<B: Backend> DepthAnything3<B> {
    pub fn new(device: &B::Device, config: DepthAnything3Config) -> Self {
        let backbone = Backbone::new(device, &config);
        let (head_mono, head_dual) = if config.head.dual_head {
            (
                None,
                Some(DualDepthAnything3Head::new(device, config.head.clone())),
            )
        } else {
            (
                Some(DepthAnything3Head::new(device, config.head.clone())),
                None,
            )
        };
        let camera_encoder = config
            .camera_encoder
            .clone()
            .map(|cfg| CameraEncoder::new(device, cfg));
        let camera_decoder = config
            .camera_decoder
            .clone()
            .map(|cfg| CameraDecoder::new(device, cfg));
        Self {
            backbone,
            head_mono,
            head_dual,
            camera_encoder,
            camera_decoder,
            img_size: Ignored(config.image_size),
            patch_size: Ignored(config.patch_size),
            hook_block_ids: Ignored(config.hook_block_ids),
            patch_token_start: Ignored(1),
        }
    }

    pub fn infer(&self, input: Tensor<B, 4>) -> DepthAnything3Inference<B> {
        let mut cache = PosEmbedCache::new();
        self.infer_with_context(input, None, None, &mut cache)
    }

    pub fn infer_with_cache(
        &self,
        input: Tensor<B, 4>,
        cache: &mut PosEmbedCache<B>,
    ) -> DepthAnything3Inference<B> {
        self.infer_with_context(input, None, None, cache)
    }

    pub fn infer_with_camera(
        &self,
        input: Tensor<B, 4>,
        extrinsics: Tensor<B, 4>,
        intrinsics: Tensor<B, 4>,
    ) -> DepthAnything3Inference<B> {
        let mut cache = PosEmbedCache::new();
        self.infer_with_context(input, Some(extrinsics), Some(intrinsics), &mut cache)
    }

    pub fn infer_with_camera_cache(
        &self,
        input: Tensor<B, 4>,
        extrinsics: Tensor<B, 4>,
        intrinsics: Tensor<B, 4>,
        cache: &mut PosEmbedCache<B>,
    ) -> DepthAnything3Inference<B> {
        self.infer_with_context(input, Some(extrinsics), Some(intrinsics), cache)
    }

    pub fn infer_with_trace(
        &self,
        input: Tensor<B, 4>,
    ) -> (DepthAnything3Inference<B>, DepthTrace<B>) {
        let mut cache = PosEmbedCache::new();
        self.infer_with_trace_cache(input, &mut cache)
    }

    pub fn infer_with_trace_cache(
        &self,
        input: Tensor<B, 4>,
        cache: &mut PosEmbedCache<B>,
    ) -> (DepthAnything3Inference<B>, DepthTrace<B>) {
        let (head_output, camera_prediction, hooks) =
            self.forward_with_camera_internal(input, None, None, cache);
        let aux_stage_necks = match &head_output {
            HeadForwardOutput::Dual(output) => Some(output.aux_stage_necks.clone()),
            _ => None,
        };
        let aux_logits = match &head_output {
            HeadForwardOutput::Dual(output) => Some(output.aux_logits.clone()),
            _ => None,
        };
        let aux_head_input = match &head_output {
            HeadForwardOutput::Dual(output) => Some(output.aux_head_input.clone()),
            _ => None,
        };
        let inference = self.finalize_inference(head_output, camera_prediction);
        let backbone_tokens = hooks
            .into_iter()
            .map(|hook| hook.patches)
            .collect::<Vec<_>>();
        (
            inference,
            DepthTrace {
                backbone_tokens,
                aux_stage_necks,
                aux_logits,
                aux_head_input,
            },
        )
    }

    pub fn infer_raw(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut cache = PosEmbedCache::new();
        self.infer_raw_with_cache(input, &mut cache)
    }

    pub fn infer_raw_with_cache(
        &self,
        input: Tensor<B, 4>,
        cache: &mut PosEmbedCache<B>,
    ) -> Tensor<B, 4> {
        match self.forward_with_camera(input, None, None, cache).0 {
            HeadForwardOutput::Mono(logits) => logits,
            HeadForwardOutput::Dual(output) => output.depth_logits,
        }
    }

    fn select_depth_channel(&self, tensor: Tensor<B, 4>) -> Tensor<B, 3> {
        if let Some(head) = &self.head_mono {
            head.select_depth_channel(tensor)
        } else {
            panic!("DepthAnything3 mono head not configured");
        }
    }

    pub fn infer_from_tokens(
        &self,
        patches: &[Tensor<B, 3>],
        height: usize,
        width: usize,
    ) -> (
        DepthAnything3Inference<B>,
        Option<Vec<Tensor<B, 4>>>,
        Option<Tensor<B, 4>>,
        Option<Tensor<B, 4>>,
    ) {
        let mut cache = PosEmbedCache::new();
        self.infer_from_tokens_with_cache(patches, height, width, &mut cache)
    }

    #[allow(clippy::type_complexity)]
    pub fn infer_from_tokens_with_cache(
        &self,
        patches: &[Tensor<B, 3>],
        height: usize,
        width: usize,
        cache: &mut PosEmbedCache<B>,
    ) -> (
        DepthAnything3Inference<B>,
        Option<Vec<Tensor<B, 4>>>,
        Option<Tensor<B, 4>>,
        Option<Tensor<B, 4>>,
    ) {
        let expected_tokens =
            (height / self.patch_size.0).max(1) * (width / self.patch_size.0).max(1);
        let patch_start = patches
            .first()
            .map(|tensor| tensor.shape().dims::<3>()[1])
            .filter(|&tokens| tokens == expected_tokens)
            .map(|_| 0)
            .unwrap_or(self.patch_token_start.0);
        let head_output = if let Some(head) = &self.head_mono {
            let inputs = patches.to_vec();
            HeadForwardOutput::Mono(head.forward_raw(
                &inputs,
                height,
                width,
                patch_start,
                self.patch_size.0,
                cache,
            ))
        } else if let Some(head) = &self.head_dual {
            let hooks = patches
                .iter()
                .cloned()
                .map(|patches| DinoIntermediate {
                    patches,
                    camera: None,
                })
                .collect::<Vec<_>>();
            HeadForwardOutput::Dual(head.forward_dual(
                &hooks,
                height,
                width,
                patch_start,
                self.patch_size.0,
                cache,
            ))
        } else {
            panic!("DepthAnything3 has no head variant configured");
        };
        let aux_stage_necks = match &head_output {
            HeadForwardOutput::Dual(output) => Some(output.aux_stage_necks.clone()),
            _ => None,
        };
        let aux_logits = match &head_output {
            HeadForwardOutput::Dual(output) => Some(output.aux_logits.clone()),
            _ => None,
        };
        let aux_head_input = match &head_output {
            HeadForwardOutput::Dual(output) => Some(output.aux_head_input.clone()),
            _ => None,
        };
        let inference = self.finalize_inference(head_output, None);
        (inference, aux_stage_necks, aux_logits, aux_head_input)
    }

    fn infer_with_context(
        &self,
        input: Tensor<B, 4>,
        extrinsics: Option<Tensor<B, 4>>,
        intrinsics: Option<Tensor<B, 4>>,
        cache: &mut PosEmbedCache<B>,
    ) -> DepthAnything3Inference<B> {
        let (head_output, camera_prediction) =
            self.forward_with_camera(input, extrinsics, intrinsics, cache);
        self.finalize_inference(head_output, camera_prediction)
    }

    fn forward_with_camera(
        &self,
        input: Tensor<B, 4>,
        extrinsics: Option<Tensor<B, 4>>,
        intrinsics: Option<Tensor<B, 4>>,
        cache: &mut PosEmbedCache<B>,
    ) -> (HeadForwardOutput<B>, Option<CameraPrediction<B>>) {
        let (head_output, camera_prediction, _) =
            self.forward_with_camera_internal(input, extrinsics, intrinsics, cache);
        (head_output, camera_prediction)
    }

    fn forward_with_camera_internal(
        &self,
        input: Tensor<B, 4>,
        extrinsics: Option<Tensor<B, 4>>,
        intrinsics: Option<Tensor<B, 4>>,
        cache: &mut PosEmbedCache<B>,
    ) -> (
        HeadForwardOutput<B>,
        Option<CameraPrediction<B>>,
        Vec<DinoIntermediate<B>>,
    ) {
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

        let camera_token = match (&self.camera_encoder, extrinsics, intrinsics) {
            (Some(encoder), Some(extr), Some(intr)) => {
                Some(encoder.forward(extr, intr, height, width))
            }
            _ => None,
        };

        let (_, hooks) =
            self.backbone
                .forward_with_hooks(input, &self.hook_block_ids.0, camera_token);
        assert!(
            hooks.len() >= self.hook_block_ids.0.len(),
            "Backbone returned fewer hooks ({}) than requested ({})",
            hooks.len(),
            self.hook_block_ids.0.len()
        );
        let patch_start = 0;
        let head_output = if let Some(head) = &self.head_mono {
            let patches: Vec<Tensor<B, 3>> =
                hooks.iter().map(|hook| hook.patches.clone()).collect();
            HeadForwardOutput::Mono(head.forward_raw(
                &patches,
                height,
                width,
                patch_start,
                self.patch_size.0,
                cache,
            ))
        } else if let Some(head) = &self.head_dual {
            HeadForwardOutput::Dual(head.forward_dual(
                &hooks,
                height,
                width,
                patch_start,
                self.patch_size.0,
                cache,
            ))
        } else {
            panic!("DepthAnything3 has no head variant configured");
        };
        let camera_prediction = self.decode_camera(&hooks, height, width);
        (head_output, camera_prediction, hooks)
    }

    pub fn img_size(&self) -> usize {
        self.img_size.0
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size.0
    }

    fn decode_camera(
        &self,
        hooks: &[DinoIntermediate<B>],
        height: usize,
        width: usize,
    ) -> Option<CameraPrediction<B>> {
        let decoder = self.camera_decoder.as_ref()?;
        let hook = hooks.last()?;
        let camera_tokens = hook.camera.as_ref()?;
        let features = camera_tokens.clone().unsqueeze_dim::<3>(1);
        Some(decoder.forward(features, None, height, width))
    }

    fn finalize_inference(
        &self,
        head_output: HeadForwardOutput<B>,
        camera_prediction: Option<CameraPrediction<B>>,
    ) -> DepthAnything3Inference<B> {
        let (pose_encoding, extrinsics, intrinsics) = if let Some(prediction) = camera_prediction {
            (
                Some(prediction.pose_encoding),
                Some(prediction.extrinsics),
                Some(prediction.intrinsics),
            )
        } else {
            (None, None, None)
        };
        match head_output {
            HeadForwardOutput::Mono(logits) => {
                let depth = self.select_depth_channel(logits);
                DepthAnything3Inference {
                    depth,
                    depth_confidence: None,
                    aux: None,
                    aux_confidence: None,
                    pose_encoding,
                    extrinsics,
                    intrinsics,
                }
            }
            HeadForwardOutput::Dual(output) => DepthAnything3Inference {
                depth: output.depth,
                depth_confidence: Some(output.depth_confidence),
                aux: Some(output.aux),
                aux_confidence: Some(output.aux_confidence),
                pose_encoding,
                extrinsics,
                intrinsics,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceBackend;
    #[cfg(feature = "backend_wgpu")]
    use burn::record::{HalfPrecisionSettings, Record};

    #[test]
    fn depth_anything3_emits_depth_tensor() {
        let device = <InferenceBackend as Backend>::Device::default();
        let config = DepthAnything3Config::metric_large();
        let model = DepthAnything3::<InferenceBackend>::new(&device, config);
        let input = Tensor::<InferenceBackend, 4>::zeros([1, 3, 518, 518], &device);
        let output = model.infer(input);
        assert_eq!(output.depth.shape().dims(), [1, 518, 518]);
    }

    #[test]
    fn cached_depth_anything3_matches_uncached() {
        let device = <InferenceBackend as Backend>::Device::default();
        let config = DepthAnything3Config::metric_large();
        let model = DepthAnything3::<InferenceBackend>::new(&device, config);
        let cached = CachedDepthAnything3::new(model.clone());

        let input_a = Tensor::<InferenceBackend, 4>::zeros([1, 3, 518, 518], &device);
        let input_b = input_a.clone();

        let base = model.infer(input_a);
        let cached_first = cached.infer(input_b.clone());
        let cached_second = cached.infer(input_b);

        let base_data = base.depth.into_data().convert::<f32>();
        let cached_first_data = cached_first.depth.into_data().convert::<f32>();
        let cached_second_data = cached_second.depth.into_data().convert::<f32>();

        assert_eq!(
            base_data.to_vec::<f32>().unwrap(),
            cached_first_data.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            cached_first_data.to_vec::<f32>().unwrap(),
            cached_second_data.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn pos_embed_cache_reused_across_inference() {
        let device = <InferenceBackend as Backend>::Device::default();
        let config = DepthAnything3Config::metric_large();
        let model = DepthAnything3::<InferenceBackend>::new(&device, config);
        let mut cache = PosEmbedCache::new();

        let input = Tensor::<InferenceBackend, 4>::zeros([1, 3, 518, 518], &device);
        assert_eq!(cache.entry_count(), 0);
        model.infer_with_cache(input.clone(), &mut cache);
        let first_count = cache.entry_count();
        assert!(first_count > 0);
        model.infer_with_cache(input, &mut cache);
        assert_eq!(first_count, cache.entry_count());
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn depth_anything3_wgpu_record_roundtrip() {
        type TestBackend = burn::backend::Wgpu<f32>;
        let device = <TestBackend as Backend>::Device::default();
        let config = DepthAnything3Config::small();
        let model = DepthAnything3::<TestBackend>::new(&device, config);
        let record_item = model
            .clone()
            .into_record()
            .into_item::<HalfPrecisionSettings>();
        let _ = <DepthAnything3<TestBackend> as Module<TestBackend>>::Record::from_item(
            record_item,
            &device,
        );
    }
}
