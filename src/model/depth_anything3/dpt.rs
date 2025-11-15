use crate::model::depth_anything3::interpolate::resize_bilinear;
use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    prelude::*,
    tensor::activation::relu,
};

#[derive(Config, Debug)]
pub struct DepthAnything3HeadConfig {
    pub dim_in: usize,
    pub features: usize,
    pub out_channels: [usize; 4],
    pub output_dim: usize,
    pub activation: HeadActivation,
    pub down_ratio: usize,
}

impl DepthAnything3HeadConfig {
    pub fn metric_large() -> Self {
        Self {
            dim_in: 1024,
            features: 256,
            out_channels: [256, 512, 1024, 1024],
            output_dim: 1,
            activation: HeadActivation::Exp,
            down_ratio: 1,
        }
    }
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
        );

        Self {
            projects,
            resize_layers,
            scratch,
            activation: Ignored(config.activation),
            down_ratio: Ignored(config.down_ratio.max(1)),
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
            resized.push(self.prepare_stage(tokens, stage, ph, pw, patch_start_idx));
        }

        let fused = self.fuse(resized);
        let fused = self.scratch.output_conv1.forward(fused);
        let target = [
            (ph * patch_size) / self.down_ratio.0,
            (pw * patch_size) / self.down_ratio.0,
        ];
        let fused = resize_bilinear(fused, target, true);
        let logits = self.scratch.output_conv2.forward(fused);
        self.apply_activation(logits).squeeze_dim(1)
    }

    fn prepare_stage(
        &self,
        tokens: Tensor<B, 3>,
        stage_idx: usize,
        ph: usize,
        pw: usize,
        patch_start_idx: usize,
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

        let x = x
            .permute([0, 2, 1])
            .reshape([batch as i32, channels as i32, ph as i32, pw as i32]);
        let x = self.projects[stage_idx].forward(x);
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
}

impl<B: Backend> Scratch<B> {
    fn new(
        device: &B::Device,
        in_channels: &[usize; 4],
        base_features: usize,
        output_dim: usize,
    ) -> Self {
        Self {
            layer1_rn: conv3x3(device, in_channels[0], base_features),
            layer2_rn: conv3x3(device, in_channels[1], base_features),
            layer3_rn: conv3x3(device, in_channels[2], base_features),
            layer4_rn: conv3x3(device, in_channels[3], base_features),
            refinenet1: FeatureFusionBlock::new(device, base_features, true),
            refinenet2: FeatureFusionBlock::new(device, base_features, true),
            refinenet3: FeatureFusionBlock::new(device, base_features, true),
            refinenet4: FeatureFusionBlock::new(device, base_features, false),
            output_conv1: Conv2dConfig::new([base_features, base_features / 2], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .init(device),
            output_conv2: ConvStack::new(device, base_features / 2, output_dim, 32),
        }
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
