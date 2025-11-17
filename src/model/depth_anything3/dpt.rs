use crate::model::depth_anything3::interpolate::resize_bilinear;
use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::{
        LayerNorm, LayerNormConfig,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::*,
    tensor::activation::relu,
};
use burn_dino::model::dino::DinoIntermediate;

#[derive(Config, Debug)]
pub struct DepthAnything3HeadConfig {
    pub dim_in: usize,
    pub features: usize,
    pub out_channels: [usize; 4],
    pub output_dim: usize,
    pub activation: HeadActivation,
    #[config(default = "HeadActivation::ExpP1")]
    pub conf_activation: HeadActivation,
    pub down_ratio: usize,
    #[config(default = "true")]
    pub pos_embed: bool,
    #[config(default = "false")]
    pub dual_head: bool,
    #[config(default = "4")]
    pub aux_levels: usize,
    #[config(default = "5")]
    pub aux_out1_conv_num: usize,
    #[config(default = "7")]
    pub aux_output_dim: usize,
    #[config(default = "true")]
    pub aux_use_layer_norm: bool,
    #[config(default = "None")]
    pub aux_layer_norm_stages: Option<Vec<usize>>,
}

impl DepthAnything3HeadConfig {
    pub fn metric_large() -> Self {
        Self {
            dim_in: 1024,
            features: 256,
            out_channels: [256, 512, 1024, 1024],
            output_dim: 1,
            activation: HeadActivation::Exp,
            conf_activation: HeadActivation::Exp,
            down_ratio: 1,
            pos_embed: true,
            dual_head: false,
            aux_levels: 4,
            aux_out1_conv_num: 5,
            aux_output_dim: 7,
            aux_use_layer_norm: true,
            aux_layer_norm_stages: None,
        }
    }

    pub fn small() -> Self {
        let aux_levels = 4;
        Self {
            dim_in: 768,
            features: 64,
            out_channels: [48, 96, 192, 384],
            output_dim: 2,
            activation: HeadActivation::Exp,
            conf_activation: HeadActivation::ExpP1,
            down_ratio: 1,
            pos_embed: true,
            dual_head: true,
            aux_levels,
            aux_out1_conv_num: 5,
            aux_output_dim: 7,
            aux_use_layer_norm: true,
            aux_layer_norm_stages: None,
        }
    }
}

fn build_layer_norm_flags(levels: usize, default: bool, custom: Option<&Vec<usize>>) -> Vec<bool> {
    let mut flags = vec![default; levels];
    if let Some(indices) = custom {
        if !indices.is_empty() {
            flags.fill(false);
            for &idx in indices {
                if idx < levels {
                    flags[idx] = true;
                }
            }
        }
    }
    flags
}

#[derive(Config, Debug)]
pub enum HeadActivation {
    Linear,
    Exp,
    ExpP1,
    ExpM1,
    Relu,
    Sigmoid,
    Softplus,
    Tanh,
}

impl Default for HeadActivation {
    fn default() -> Self {
        HeadActivation::Linear
    }
}

#[derive(Module, Debug)]
pub struct DepthAnything3Head<B: Backend> {
    projects: Vec<Conv2d<B>>,
    resize_layers: Vec<ResizeOp<B>>,
    scratch: Scratch<B>,
    activation: Ignored<HeadActivation>,
    down_ratio: Ignored<usize>,
    project_input_dim: Ignored<usize>,
    pos_embed: bool,
}

#[derive(Module, Debug)]
pub struct DualDepthAnything3Head<B: Backend> {
    norm: LayerNorm<B>,
    projects: Vec<Conv2d<B>>,
    resize_layers: Vec<ResizeOp<B>>,
    scratch: Scratch<B>,
    activation: Ignored<HeadActivation>,
    conf_activation: Ignored<HeadActivation>,
    down_ratio: Ignored<usize>,
    pos_embed: bool,
    aux_levels: Ignored<usize>,
    aux_output_dim: Ignored<usize>,
    project_input_dim: Ignored<usize>,
}

pub struct DualHeadOutput<B: Backend> {
    pub depth_logits: Tensor<B, 4>,
    pub depth: Tensor<B, 3>,
    pub depth_confidence: Tensor<B, 3>,
    pub aux_logits: Tensor<B, 4>,
    pub aux: Tensor<B, 4>,
    pub aux_confidence: Tensor<B, 3>,
    pub aux_stage_necks: Vec<Tensor<B, 4>>,
    pub aux_head_input: Tensor<B, 4>,
}

impl<B: Backend> DualDepthAnything3Head<B> {
    pub fn new(device: &B::Device, config: DepthAnything3HeadConfig) -> Self {
        let mut projects = Vec::with_capacity(4);
        for &channels in &config.out_channels {
            projects.push(
                Conv2dConfig::new([config.dim_in, channels], [1, 1])
                    .with_bias(true)
                    .init(device),
            );
        }

        let resize_layers = vec![
            ResizeOp::conv_transpose(
                ConvTranspose2dConfig::new(
                    [config.out_channels[0], config.out_channels[0]],
                    [4, 4],
                )
                .with_stride([4, 4])
                .with_bias(true)
                .init(device),
            ),
            ResizeOp::conv_transpose(
                ConvTranspose2dConfig::new(
                    [config.out_channels[1], config.out_channels[1]],
                    [2, 2],
                )
                .with_stride([2, 2])
                .with_bias(true)
                .init(device),
            ),
            ResizeOp::identity(),
            ResizeOp::conv(
                Conv2dConfig::new([config.out_channels[3], config.out_channels[3]], [3, 3])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                    .with_stride([2, 2])
                    .with_bias(true)
                    .init(device),
            ),
        ];

        let aux_config = ScratchAuxConfig {
            levels: config.aux_levels.max(1),
            out1_conv_num: config.aux_out1_conv_num.max(1),
            output_dim: config.aux_output_dim.max(2),
            layer_norm_flags: build_layer_norm_flags(
                config.aux_levels.max(1),
                config.aux_use_layer_norm,
                config.aux_layer_norm_stages.as_ref(),
            ),
        };

        let scratch = Scratch::new(
            device,
            &config.out_channels,
            config.features,
            config.output_dim,
            Some(aux_config),
        );

        Self {
            norm: LayerNormConfig::new(config.dim_in).init(device),
            projects,
            resize_layers,
            scratch,
            activation: Ignored(config.activation),
            conf_activation: Ignored(config.conf_activation),
            down_ratio: Ignored(config.down_ratio.max(1)),
            pos_embed: config.pos_embed,
            aux_levels: Ignored(config.aux_levels.max(1)),
            aux_output_dim: Ignored(config.aux_output_dim.max(2)),
            project_input_dim: Ignored(config.dim_in),
        }
    }

    pub fn forward_dual(
        &self,
        hooks: &[DinoIntermediate<B>],
        height: usize,
        width: usize,
        patch_start_idx: usize,
        patch_size: usize,
    ) -> DualHeadOutput<B> {
        assert!(
            hooks.len() >= 4,
            "DualDepthAnything3Head expects at least 4 hook tensors, got {}",
            hooks.len()
        );
        let ph = height / patch_size;
        let pw = width / patch_size;

        let mut resized = Vec::with_capacity(4);
        for stage in 0..4 {
            let tokens = hooks[stage].patches.clone();
            resized.push(self.prepare_stage(tokens, stage, ph, pw, patch_start_idx, height, width));
        }

        let fused_main = self.fuse_main(&resized);
        let main_logits = self.build_main_logits(fused_main, height, width);

        let (aux_logits, aux_stage_necks, aux_head_input) =
            self.build_aux_logits(&resized, height, width);

        let depth = self.select_main_channel(&main_logits, 0);
        let depth_confidence = self.select_conf_channel(&main_logits);
        let aux = self.select_aux_values(&aux_logits);
        let aux_confidence = self.select_aux_confidence(&aux_logits);

        DualHeadOutput {
            depth_logits: main_logits,
            depth,
            depth_confidence,
            aux_logits,
            aux,
            aux_confidence,
            aux_stage_necks,
            aux_head_input,
        }
    }

    fn prepare_stage(
        &self,
        tokens: Tensor<B, 3>,
        stage: usize,
        ph: usize,
        pw: usize,
        patch_start_idx: usize,
        height: usize,
        width: usize,
    ) -> Tensor<B, 4> {
        let dims = tokens.shape().dims::<3>();
        let batch = dims[0];
        let mut selected = tokens;
        if patch_start_idx > 0 {
            selected = selected.slice([
                0..batch as i32,
                patch_start_idx as i32..dims[1] as i32,
                0..dims[2] as i32,
            ]);
        }
        let normalized = self.norm.forward(selected);
        let projected = self.projects[stage].forward(normalized.permute([0, 2, 1]).reshape([
            batch as i32,
            self.project_input_dim.0 as i32,
            ph as i32,
            pw as i32,
        ]));

        let mut projected = projected;
        if self.pos_embed {
            projected = add_position_embedding(projected, width, height);
        }
        self.resize_layers[stage].forward(projected)
    }

    fn fuse_main(&self, features: &[Tensor<B, 4>]) -> Tensor<B, 4> {
        let l1 = self.scratch.layer1_rn.forward(features[0].clone());
        let l2 = self.scratch.layer2_rn.forward(features[1].clone());
        let l3 = self.scratch.layer3_rn.forward(features[2].clone());
        let l4 = self.scratch.layer4_rn.forward(features[3].clone());

        let mut out = self.scratch.refinenet4.forward(l4, None, Some(hw(&l3)));
        out = self
            .scratch
            .refinenet3
            .forward(out, Some(l3), Some(hw(&l2)));
        out = self
            .scratch
            .refinenet2
            .forward(out, Some(l2), Some(hw(&l1)));
        self.scratch.refinenet1.forward(out, Some(l1), None)
    }

    fn build_main_logits(&self, fused: Tensor<B, 4>, height: usize, width: usize) -> Tensor<B, 4> {
        let mut fused = self.scratch.output_conv1.forward(fused);
        let target = [
            (height / self.down_ratio.0).max(1),
            (width / self.down_ratio.0).max(1),
        ];
        fused = resize_bilinear(fused, target, true);
        if self.pos_embed {
            fused = add_position_embedding(fused, width, height);
        }
        self.scratch.output_conv2.forward(fused)
    }

    fn build_aux_logits(
        &self,
        features: &[Tensor<B, 4>],
        height: usize,
        width: usize,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>, Tensor<B, 4>) {
        let aux1 = self
            .scratch
            .refinenet1_aux
            .as_ref()
            .expect("aux refinenet1 missing");
        let aux2 = self
            .scratch
            .refinenet2_aux
            .as_ref()
            .expect("aux refinenet2 missing");
        let aux3 = self
            .scratch
            .refinenet3_aux
            .as_ref()
            .expect("aux refinenet3 missing");
        let aux4 = self
            .scratch
            .refinenet4_aux
            .as_ref()
            .expect("aux refinenet4 missing");
        let output_conv1_aux = self
            .scratch
            .output_conv1_aux
            .as_ref()
            .expect("aux pre heads missing");
        let output_conv2_aux = self
            .scratch
            .output_conv2_aux
            .as_ref()
            .expect("aux output heads missing");

        let l1 = self.scratch.layer1_rn.forward(features[0].clone());
        let l2 = self.scratch.layer2_rn.forward(features[1].clone());
        let l3 = self.scratch.layer3_rn.forward(features[2].clone());
        let l4 = self.scratch.layer4_rn.forward(features[3].clone());

        let mut aux_out = aux4.forward(l4.clone(), None, Some(hw(&l3)));
        let mut aux_levels = Vec::with_capacity(self.aux_levels.0);
        if self.aux_levels.0 >= 4 {
            aux_levels.push(aux_out.clone());
        }
        aux_out = aux3.forward(aux_out, Some(l3), Some(hw(&l2)));
        if self.aux_levels.0 >= 3 {
            aux_levels.push(aux_out.clone());
        }
        aux_out = aux2.forward(aux_out, Some(l2), Some(hw(&l1)));
        if self.aux_levels.0 >= 2 {
            aux_levels.push(aux_out.clone());
        }
        aux_out = aux1.forward(aux_out, Some(l1.clone()), None);
        aux_levels.push(aux_out);

        let mut processed = Vec::with_capacity(aux_levels.len());
        for (idx, aux) in aux_levels.iter().enumerate() {
            let neck = output_conv1_aux
                .get(idx)
                .expect("missing aux neck")
                .forward(aux.clone());
            processed.push(neck);
        }

        let mut last = processed
            .last()
            .cloned()
            .expect("aux levels must not be empty");
        if self.pos_embed {
            last = add_position_embedding(last, width, height);
        }
        let head_input = if self.pos_embed {
            add_position_embedding(last.clone(), width, height)
        } else {
            last.clone()
        };
        let logits = output_conv2_aux
            .last()
            .expect("missing aux output head")
            .forward(head_input.clone());
        (logits, processed, head_input)
    }

    fn select_main_channel(&self, tensor: &Tensor<B, 4>, index: usize) -> Tensor<B, 3> {
        let dims = tensor.shape().dims::<4>();
        let channel = tensor
            .clone()
            .slice([
                0..dims[0] as i32,
                index as i32..index as i32 + 1,
                0..dims[2] as i32,
                0..dims[3] as i32,
            ])
            .squeeze_dim(1);
        self.activate_scalar(channel, self.activation.0.clone())
    }

    fn select_conf_channel(&self, tensor: &Tensor<B, 4>) -> Tensor<B, 3> {
        let dims = tensor.shape().dims::<4>();
        let channel = tensor
            .clone()
            .slice([
                0..dims[0] as i32,
                (dims[1] - 1) as i32..dims[1] as i32,
                0..dims[2] as i32,
                0..dims[3] as i32,
            ])
            .squeeze_dim(1);
        self.activate_scalar(channel, self.conf_activation.0.clone())
    }

    fn select_aux_values(&self, tensor: &Tensor<B, 4>) -> Tensor<B, 4> {
        let dims = tensor.shape().dims::<4>();
        tensor.clone().slice([
            0..dims[0] as i32,
            0..(self.aux_output_dim.0 - 1) as i32,
            0..dims[2] as i32,
            0..dims[3] as i32,
        ])
    }

    fn select_aux_confidence(&self, tensor: &Tensor<B, 4>) -> Tensor<B, 3> {
        let dims = tensor.shape().dims::<4>();
        let channel = tensor
            .clone()
            .slice([
                0..dims[0] as i32,
                (self.aux_output_dim.0 - 1) as i32..self.aux_output_dim.0 as i32,
                0..dims[2] as i32,
                0..dims[3] as i32,
            ])
            .squeeze_dim(1);
        self.activate_scalar(channel, self.conf_activation.0.clone())
    }

    fn activate_scalar(&self, tensor: Tensor<B, 3>, kind: HeadActivation) -> Tensor<B, 3> {
        match kind {
            HeadActivation::Linear => tensor,
            HeadActivation::Exp => tensor.exp(),
            HeadActivation::ExpP1 => tensor.exp().add_scalar(1.0),
            HeadActivation::ExpM1 => tensor.exp().add_scalar(-1.0),
            HeadActivation::Relu => relu(tensor),
            HeadActivation::Sigmoid => {
                sigmoid_tensor(tensor.clone().unsqueeze_dim::<4>(1)).squeeze_dim::<3>(1)
            }
            HeadActivation::Softplus => {
                (tensor.clone().unsqueeze_dim::<4>(1).exp().add_scalar(1.0))
                    .log()
                    .squeeze_dim::<3>(1)
            }
            HeadActivation::Tanh => tanh_tensor(tensor.unsqueeze_dim::<4>(1)).squeeze_dim::<3>(1),
        }
    }
}

impl<B: Backend> DepthAnything3Head<B> {
    pub fn new(device: &B::Device, config: DepthAnything3HeadConfig) -> Self {
        let mut projects = Vec::with_capacity(4);
        for &channels in &config.out_channels {
            projects.push(
                Conv2dConfig::new([config.dim_in, channels], [1, 1])
                    .with_bias(true)
                    .init(device),
            );
        }

        let resize_layers = vec![
            ResizeOp::conv_transpose(
                ConvTranspose2dConfig::new(
                    [config.out_channels[0], config.out_channels[0]],
                    [4, 4],
                )
                .with_stride([4, 4])
                .with_bias(true)
                .init(device),
            ),
            ResizeOp::conv_transpose(
                ConvTranspose2dConfig::new(
                    [config.out_channels[1], config.out_channels[1]],
                    [2, 2],
                )
                .with_stride([2, 2])
                .with_bias(true)
                .init(device),
            ),
            ResizeOp::identity(),
            ResizeOp::conv(
                Conv2dConfig::new([config.out_channels[3], config.out_channels[3]], [3, 3])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                    .with_stride([2, 2])
                    .with_bias(true)
                    .init(device),
            ),
        ];

        let scratch = Scratch::new(
            device,
            &config.out_channels,
            config.features,
            config.output_dim,
            None,
        );
        Self {
            projects,
            resize_layers,
            scratch,
            activation: Ignored(config.activation),
            down_ratio: Ignored(config.down_ratio.max(1)),
            project_input_dim: Ignored(config.dim_in),
            pos_embed: config.pos_embed,
        }
    }

    pub fn forward(
        &self,
        hooks: &[Tensor<B, 3>],
        height: usize,
        width: usize,
        patch_start_idx: usize,
        patch_size: usize,
    ) -> Tensor<B, 3> {
        let activated = self.forward_raw(hooks, height, width, patch_start_idx, patch_size);
        self.select_depth_channel(activated)
    }

    pub fn forward_raw(
        &self,
        hooks: &[Tensor<B, 3>],
        height: usize,
        width: usize,
        patch_start_idx: usize,
        patch_size: usize,
    ) -> Tensor<B, 4> {
        assert!(
            hooks.len() >= 4,
            "DepthAnything3Head expects at least 4 hook tensors, got {}",
            hooks.len()
        );
        let ph = height / patch_size;
        let pw = width / patch_size;

        let mut resized = Vec::with_capacity(4);
        for stage in 0..4 {
            let tokens = hooks[stage].clone();
            resized.push(self.prepare_stage(tokens, stage, ph, pw, patch_start_idx, height, width));
        }

        let fused = self.fuse(resized);
        let fused = self.scratch.output_conv1.forward(fused);
        let target = [
            (ph * patch_size) / self.down_ratio.0,
            (pw * patch_size) / self.down_ratio.0,
        ];
        let mut fused = resize_bilinear(fused, target, true);
        if self.pos_embed {
            fused = self.add_pos_embed(fused, width, height);
        }
        let logits = self.scratch.output_conv2.forward(fused);
        self.apply_activation(logits)
    }

    pub fn select_depth_channel(&self, tensor: Tensor<B, 4>) -> Tensor<B, 3> {
        let dims = tensor.shape().dims::<4>();
        if dims[1] == 1 {
            tensor.squeeze_dim(1)
        } else {
            tensor
                .slice([
                    0..dims[0] as i32,
                    0..1,
                    0..dims[2] as i32,
                    0..dims[3] as i32,
                ])
                .squeeze_dim(1)
        }
    }

    fn prepare_stage(
        &self,
        tokens: Tensor<B, 3>,
        stage_idx: usize,
        ph: usize,
        pw: usize,
        patch_start_idx: usize,
        image_height: usize,
        image_width: usize,
    ) -> Tensor<B, 4> {
        let dims = tokens.shape().dims::<3>();
        let batch = dims[0];
        let tokens_per_stage = dims[1];
        let channels = dims[2];
        let patch_tokens = ph * pw;
        let end_idx = patch_start_idx + patch_tokens;

        assert!(
            tokens_per_stage >= end_idx,
            "Hook tensor for stage {stage_idx} does not have enough patch tokens"
        );
        debug_assert_eq!(
            tokens_per_stage,
            patch_start_idx + patch_tokens,
            "Unexpected token count at stage {stage_idx}"
        );

        let x = tokens.slice([0..batch, patch_start_idx..end_idx, 0..channels]);
        let x = self.apply_token_norm(x);

        let x = x
            .permute([0, 2, 1])
            .reshape([batch as i32, channels as i32, ph as i32, pw as i32]);
        let x = ensure_channels(x, self.project_input_dim.0);
        let mut x = self.projects[stage_idx].forward(x);
        if self.pos_embed {
            x = self.add_pos_embed(x, image_width, image_height);
        }
        self.resize_layers[stage_idx].forward(x)
    }

    fn fuse(&self, feats: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
        let mut it = feats.into_iter();
        let l1 = it.next().expect("missing l1");
        let l2 = it.next().expect("missing l2");
        let l3 = it.next().expect("missing l3");
        let l4 = it.next().expect("missing l4");

        let l1_rn = self.scratch.layer1_rn.forward(l1);
        let l2_rn = self.scratch.layer2_rn.forward(l2);
        let l3_rn = self.scratch.layer3_rn.forward(l3);
        let l4_rn = self.scratch.layer4_rn.forward(l4);

        let mut out = self
            .scratch
            .refinenet4
            .forward(l4_rn.clone(), None, Some(hw(&l3_rn)));
        out = self
            .scratch
            .refinenet3
            .forward(out, Some(l3_rn.clone()), Some(hw(&l2_rn)));
        out = self
            .scratch
            .refinenet2
            .forward(out, Some(l2_rn.clone()), Some(hw(&l1_rn)));
        self.scratch.refinenet1.forward(out, Some(l1_rn), None)
    }

    fn apply_activation(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        match self.activation.0 {
            HeadActivation::Linear => tensor,
            HeadActivation::Exp => tensor.exp(),
            HeadActivation::ExpP1 => tensor.exp().add_scalar(1.0),
            HeadActivation::ExpM1 => tensor.exp().add_scalar(-1.0),
            HeadActivation::Relu => relu(tensor),
            HeadActivation::Sigmoid => sigmoid_tensor(tensor),
            HeadActivation::Softplus => (tensor.exp().add_scalar(1.0)).log(),
            HeadActivation::Tanh => tanh_tensor(tensor),
        }
    }
}

fn hw<B: Backend>(tensor: &Tensor<B, 4>) -> [usize; 2] {
    let dims = tensor.shape().dims::<4>();
    [dims[2], dims[3]]
}

fn ensure_channels<B: Backend>(input: Tensor<B, 4>, desired: usize) -> Tensor<B, 4> {
    let dims = input.shape().dims::<4>();
    let current = dims[1];
    if current == desired {
        return input;
    }

    if current > desired {
        return input.slice([
            0..dims[0] as i32,
            0..desired as i32,
            0..dims[2] as i32,
            0..dims[3] as i32,
        ]);
    }

    let pad = desired - current;
    let device = input.device();
    let zeros = Tensor::<B, 4>::zeros(
        [dims[0] as i32, pad as i32, dims[2] as i32, dims[3] as i32],
        &device,
    );

    Tensor::cat(vec![input, zeros], 1)
}

impl<B: Backend> DepthAnything3Head<B> {
    fn apply_token_norm(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let (var, mean) = tokens.clone().var_mean_bias(2);
        tokens.sub(mean).div(var.add_scalar(TOKEN_NORM_EPS).sqrt())
    }

    fn add_pos_embed(
        &self,
        tensor: Tensor<B, 4>,
        image_width: usize,
        image_height: usize,
    ) -> Tensor<B, 4> {
        if !self.pos_embed {
            return tensor;
        }
        add_position_embedding(tensor, image_width, image_height)
    }
}

const TOKEN_NORM_EPS: f32 = 1e-5;
const POS_EMBED_RATIO: f32 = 0.1;
const POS_EMBED_OMEGA0: f32 = 100.0;

fn build_positional_embedding(
    channels: usize,
    height: usize,
    width: usize,
    image_width: usize,
    image_height: usize,
) -> Vec<f32> {
    if channels == 0 {
        return vec![];
    }
    let aspect_ratio = image_width as f32 / image_height as f32;
    let diag_factor = (aspect_ratio * aspect_ratio + 1.0).sqrt();
    let span_x = aspect_ratio / diag_factor;
    let span_y = 1.0 / diag_factor;

    let left_x = -span_x * (width as f32 - 1.0) / width as f32;
    let right_x = span_x * (width as f32 - 1.0) / width as f32;
    let top_y = -span_y * (height as f32 - 1.0) / height as f32;
    let bottom_y = span_y * (height as f32 - 1.0) / height as f32;

    let x_coords = linspace(left_x, right_x, width);
    let y_coords = linspace(top_y, bottom_y, height);

    let mut chw = vec![0.0f32; height * width * channels];
    let x_channels = channels / 2;
    let y_channels = channels - x_channels;

    let embed_x_table: Vec<Vec<f32>> = x_coords
        .iter()
        .map(|&x| make_sincos_embedding(x_channels, x))
        .collect();
    let embed_y_table: Vec<Vec<f32>> = y_coords
        .iter()
        .map(|&y| make_sincos_embedding(y_channels, y))
        .collect();

    for (x_idx, embed_x) in embed_x_table.iter().enumerate() {
        for (y_idx, embed_y) in embed_y_table.iter().enumerate() {
            let pixel_index = x_idx * height + y_idx;
            for (i, &value) in embed_x.iter().enumerate() {
                let channel = i;
                let dst = channel * height * width + pixel_index;
                chw[dst] = value;
            }
            for (i, &value) in embed_y.iter().enumerate() {
                let channel = x_channels + i;
                let dst = channel * height * width + pixel_index;
                chw[dst] = value;
            }
        }
    }

    chw
}

fn add_position_embedding<B: Backend>(
    tensor: Tensor<B, 4>,
    image_width: usize,
    image_height: usize,
) -> Tensor<B, 4> {
    let dims = tensor.shape().dims::<4>();
    let batch = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];
    let embedding = build_positional_embedding(channels, height, width, image_width, image_height);
    let mut repeated = Vec::with_capacity(embedding.len() * batch);
    for _ in 0..batch {
        repeated.extend_from_slice(&embedding);
    }
    let device = tensor.device();
    let embed_tensor = Tensor::<B, 1>::from_floats(repeated.as_slice(), &device).reshape([
        batch as i32,
        channels as i32,
        height as i32,
        width as i32,
    ]);
    let expanded = embed_tensor.expand(tensor.shape());
    tensor + expanded.mul_scalar(POS_EMBED_RATIO)
}

fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps <= 1 {
        return vec![start];
    }
    let step = (end - start) / (steps as f32 - 1.0);
    (0..steps).map(|i| start + step * i as f32).collect()
}

fn make_sincos_embedding(dim: usize, position: f32) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let half = dim / 2;
    let mut values = Vec::with_capacity(dim);
    for i in 0..half {
        let exponent = if half > 0 {
            i as f32 / half as f32
        } else {
            0.0
        };
        let omega = POS_EMBED_OMEGA0.powf(-exponent);
        let input = position * omega;
        values.push(input.sin());
    }
    let remaining = dim - half;
    for i in 0..remaining {
        let exponent = if remaining > 0 {
            i as f32 / remaining as f32
        } else {
            0.0
        };
        let omega = POS_EMBED_OMEGA0.powf(-exponent);
        let input = position * omega;
        values.push(input.cos());
    }
    values
}

#[derive(Module, Debug)]
struct ResizeOp<B: Backend> {
    conv_t: Option<ConvTranspose2d<B>>,
    conv: Option<Conv2d<B>>,
}

impl<B: Backend> ResizeOp<B> {
    fn identity() -> Self {
        Self {
            conv_t: None,
            conv: None,
        }
    }

    fn conv_transpose(layer: ConvTranspose2d<B>) -> Self {
        Self {
            conv_t: Some(layer),
            conv: None,
        }
    }

    fn conv(layer: Conv2d<B>) -> Self {
        Self {
            conv_t: None,
            conv: Some(layer),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if let Some(layer) = &self.conv_t {
            layer.forward(x)
        } else if let Some(layer) = &self.conv {
            layer.forward(x)
        } else {
            x
        }
    }
}

#[derive(Module, Debug, Clone)]
struct ScratchAuxConfig {
    levels: usize,
    out1_conv_num: usize,
    output_dim: usize,
    layer_norm_flags: Vec<bool>,
}

#[derive(Module, Debug)]
struct Scratch<B: Backend> {
    layer1_rn: Conv2d<B>,
    layer2_rn: Conv2d<B>,
    layer3_rn: Conv2d<B>,
    layer4_rn: Conv2d<B>,
    refinenet1: FeatureFusionBlock<B>,
    refinenet2: FeatureFusionBlock<B>,
    refinenet3: FeatureFusionBlock<B>,
    refinenet4: FeatureFusionBlock<B>,
    output_conv1: Conv2d<B>,
    output_conv2: ConvStack<B>,
    refinenet1_aux: Option<FeatureFusionBlock<B>>,
    refinenet2_aux: Option<FeatureFusionBlock<B>>,
    refinenet3_aux: Option<FeatureFusionBlock<B>>,
    refinenet4_aux: Option<FeatureFusionBlock<B>>,
    output_conv1_aux: Option<Vec<AuxPreHead<B>>>,
    output_conv2_aux: Option<Vec<AuxOutputHead<B>>>,
    aux_levels: usize,
}

impl<B: Backend> Scratch<B> {
    fn new(
        device: &B::Device,
        in_channels: &[usize; 4],
        base_features: usize,
        output_dim: usize,
        aux_config: Option<ScratchAuxConfig>,
    ) -> Self {
        let refinenet1 = FeatureFusionBlock::new(device, base_features, true);
        let refinenet2 = FeatureFusionBlock::new(device, base_features, true);
        let refinenet3 = FeatureFusionBlock::new(device, base_features, true);
        let refinenet4 = FeatureFusionBlock::new(device, base_features, false);

        let (
            refinenet1_aux,
            refinenet2_aux,
            refinenet3_aux,
            refinenet4_aux,
            output_conv1_aux,
            output_conv2_aux,
            aux_levels,
        ) = if let Some(config) = aux_config {
            let aux1 = FeatureFusionBlock::new(device, base_features, true);
            let aux2 = FeatureFusionBlock::new(device, base_features, true);
            let aux3 = FeatureFusionBlock::new(device, base_features, true);
            let aux4 = FeatureFusionBlock::new(device, base_features, false);
            let mut pre_heads = Vec::with_capacity(config.levels);
            for _ in 0..config.levels {
                pre_heads.push(AuxPreHead::new(device, base_features, config.out1_conv_num));
            }

            let mut output_heads = Vec::with_capacity(config.levels);
            for idx in 0..config.levels {
                let use_ln = config.layer_norm_flags.get(idx).copied().unwrap_or(false);
                output_heads.push(AuxOutputHead::new(
                    device,
                    base_features / 2,
                    32,
                    config.output_dim,
                    use_ln,
                ));
            }

            (
                Some(aux1),
                Some(aux2),
                Some(aux3),
                Some(aux4),
                Some(pre_heads),
                Some(output_heads),
                config.levels,
            )
        } else {
            (None, None, None, None, None, None, 0)
        };

        Self {
            layer1_rn: conv3x3(device, in_channels[0], base_features),
            layer2_rn: conv3x3(device, in_channels[1], base_features),
            layer3_rn: conv3x3(device, in_channels[2], base_features),
            layer4_rn: conv3x3(device, in_channels[3], base_features),
            refinenet1,
            refinenet2,
            refinenet3,
            refinenet4,
            output_conv1: Conv2dConfig::new([base_features, base_features / 2], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .init(device),
            output_conv2: ConvStack::new(device, base_features / 2, output_dim, 32),
            refinenet1_aux,
            refinenet2_aux,
            refinenet3_aux,
            refinenet4_aux,
            output_conv1_aux,
            output_conv2_aux,
            aux_levels,
        }
    }
}

#[derive(Module, Debug)]
struct AuxPreHead<B: Backend> {
    layers: Vec<Conv2d<B>>,
}

impl<B: Backend> AuxPreHead<B> {
    fn new(device: &B::Device, channels: usize, count: usize) -> Self {
        let mut layers = Vec::with_capacity(count);
        let mut in_ch = channels;
        for idx in 0..count {
            let out_ch = if idx % 2 == 0 { channels / 2 } else { channels };
            layers.push(
                Conv2dConfig::new([in_ch, out_ch], [3, 3])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                    .with_bias(true)
                    .init(device),
            );
            in_ch = out_ch;
        }
        Self { layers }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = input;
        for conv in &self.layers {
            x = conv.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
struct LayerNorm2d<B: Backend> {
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> LayerNorm2d<B> {
    fn new(device: &B::Device, channels: usize) -> Self {
        let layer_norm: LayerNorm<B> = LayerNormConfig::new(channels).init(device);
        Self { layer_norm }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let dims = input.shape().dims::<4>();
        let reshaped = input
            .clone()
            .permute([0, 2, 3, 1])
            .reshape([(dims[0] * dims[2] * dims[3]) as i32, dims[1] as i32]);
        let normalized = self.layer_norm.forward(reshaped);
        normalized
            .reshape([
                dims[0] as i32,
                dims[2] as i32,
                dims[3] as i32,
                dims[1] as i32,
            ])
            .permute([0, 3, 1, 2])
    }
}

#[derive(Module, Debug)]
struct AuxOutputHead<B: Backend> {
    reduce: Conv2d<B>,
    norm: Option<LayerNorm2d<B>>,
    project: Conv2d<B>,
}

impl<B: Backend> AuxOutputHead<B> {
    fn new(
        device: &B::Device,
        in_channels: usize,
        mid_channels: usize,
        out_channels: usize,
        use_layer_norm: bool,
    ) -> Self {
        let reduce = Conv2dConfig::new([in_channels, mid_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .with_bias(true)
            .init(device);
        let norm = if use_layer_norm {
            Some(LayerNorm2d::new(device, mid_channels))
        } else {
            None
        };
        let project = Conv2dConfig::new([mid_channels, out_channels], [1, 1])
            .with_bias(true)
            .init(device);

        Self {
            reduce,
            norm,
            project,
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.reduce.forward(input);
        if let Some(norm) = &self.norm {
            x = norm.forward(x);
        }
        x = relu(x);
        self.project.forward(x)
    }
}

#[derive(Module, Debug)]
struct FeatureFusionBlock<B: Backend> {
    residual1: Option<ResidualConvUnit<B>>,
    residual2: ResidualConvUnit<B>,
    out_conv: Conv2d<B>,
}

impl<B: Backend> FeatureFusionBlock<B> {
    fn new(device: &B::Device, channels: usize, has_residual: bool) -> Self {
        let residual1 = if has_residual {
            Some(ResidualConvUnit::new(device, channels))
        } else {
            None
        };
        Self {
            residual1,
            residual2: ResidualConvUnit::new(device, channels),
            out_conv: Conv2dConfig::new([channels, channels], [1, 1])
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(
        &self,
        top: Tensor<B, 4>,
        lateral: Option<Tensor<B, 4>>,
        size: Option<[usize; 2]>,
    ) -> Tensor<B, 4> {
        let mut y = top;
        if let (Some(residual), Some(lat)) = (&self.residual1, lateral) {
            y = y + residual.forward(lat);
        }

        y = self.residual2.forward(y);
        let current_hw = hw(&y);
        let target = size.unwrap_or([current_hw[0] * 2, current_hw[1] * 2]);
        y = resize_bilinear(y, target, true);
        self.out_conv.forward(y)
    }
}

#[derive(Module, Debug)]
struct ResidualConvUnit<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> ResidualConvUnit<B> {
    fn new(device: &B::Device, channels: usize) -> Self {
        let conv = |in_ch, out_ch| {
            Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .init(device)
        };

        Self {
            conv1: conv(channels, channels),
            conv2: conv(channels, channels),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(relu(input.clone()));
        let x = self.conv2.forward(relu(x));
        x + input
    }
}

#[derive(Module, Debug)]
struct ConvStack<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> ConvStack<B> {
    fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        mid_channels: usize,
    ) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, mid_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .with_bias(true)
            .init(device);
        let conv2 = Conv2dConfig::new([mid_channels, out_channels], [1, 1])
            .with_bias(true)
            .init(device);
        Self { conv1, conv2 }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.conv1.forward(input));
        self.conv2.forward(x)
    }
}

fn conv3x3<B: Backend>(device: &B::Device, in_channels: usize, out_channels: usize) -> Conv2d<B> {
    Conv2dConfig::new([in_channels, out_channels], [3, 3])
        .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
        .with_bias(false)
        .init(device)
}

fn sigmoid_tensor<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    tensor
        .clone()
        .mul_scalar(-1.0)
        .exp()
        .add_scalar(1.0)
        .recip()
}

fn tanh_tensor<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let exp = tensor.clone().mul_scalar(2.0).exp();
    let ones = exp.clone().ones_like();
    let numerator = exp.clone() - ones.clone();
    numerator / (exp + ones)
}
