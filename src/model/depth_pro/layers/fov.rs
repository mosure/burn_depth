use burn::{
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
    },
    prelude::*,
};

use crate::model::dino::DinoVisionTransformer;
use burn::tensor::activation::relu;

#[derive(Module, Debug)]
struct ConvActivation<B: Backend> {
    conv: Conv2d<B>,
    with_relu: bool,
}

impl<B: Backend> ConvActivation<B> {
    fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        with_relu: bool,
    ) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .init(device);

        Self { conv, with_relu }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(x);
        if self.with_relu { relu(out) } else { out }
    }
}

#[derive(Module, Debug)]
pub struct FOVNetwork<B: Backend> {
    num_features: usize,
    encoder: Option<DinoVisionTransformer<B>>,
    encoder_proj: Option<Linear<B>>,
    downsample_input: Option<Interpolate2d>,
    downsample_blocks: Vec<ConvActivation<B>>,
    head_blocks: Vec<ConvActivation<B>>,
}

impl<B: Backend> FOVNetwork<B> {
    pub fn new(
        device: &B::Device,
        num_features: usize,
        fov_encoder: Option<DinoVisionTransformer<B>>,
    ) -> Self {
        let mut downsample_blocks = Vec::new();
        let mut head_blocks = Vec::new();
        let mut encoder_proj = None;
        let mut downsample_input = None;
        let mut encoder = None;

        if let Some(model) = fov_encoder {
            let embed_dim = model.embedding_dimension();
            encoder_proj = Some(LinearConfig::new(embed_dim, num_features / 2).init(device));
            downsample_input = Some(
                Interpolate2dConfig::new()
                    .with_scale_factor(Some([0.25, 0.25]))
                    .with_mode(InterpolateMode::Linear)
                    .init(),
            );

            downsample_blocks.push(ConvActivation::new(
                device,
                num_features,
                num_features / 2,
                3,
                2,
                1,
                true,
            ));

            head_blocks.push(ConvActivation::new(
                device,
                num_features / 2,
                num_features / 4,
                3,
                2,
                1,
                true,
            ));
            head_blocks.push(ConvActivation::new(
                device,
                num_features / 4,
                num_features / 8,
                3,
                2,
                1,
                true,
            ));
            head_blocks.push(ConvActivation::new(
                device,
                num_features / 8,
                1,
                6,
                1,
                0,
                false,
            ));

            encoder = Some(model);
        } else {
            head_blocks.push(ConvActivation::new(
                device,
                num_features,
                num_features / 2,
                3,
                2,
                1,
                true,
            ));
            head_blocks.push(ConvActivation::new(
                device,
                num_features / 2,
                num_features / 4,
                3,
                2,
                1,
                true,
            ));
            head_blocks.push(ConvActivation::new(
                device,
                num_features / 4,
                num_features / 8,
                3,
                2,
                1,
                true,
            ));
            head_blocks.push(ConvActivation::new(
                device,
                num_features / 8,
                1,
                6,
                1,
                0,
                false,
            ));
        }

        Self {
            num_features,
            encoder,
            encoder_proj,
            downsample_input,
            downsample_blocks,
            head_blocks,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, lowres_feature: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.encoder.is_some() {
            self.forward_with_encoder(x, lowres_feature)
        } else {
            self.apply_blocks(&self.head_blocks, lowres_feature)
        }
    }

    fn forward_with_encoder(&self, x: Tensor<B, 4>, lowres_feature: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut features = lowres_feature;
        features = self.apply_blocks(&self.downsample_blocks, features);

        let target_shape = features.shape().dims();
        let encoded = self.encode_image(x, target_shape);

        let fused = features + encoded;
        self.apply_blocks(&self.head_blocks, fused)
    }

    fn encode_image(&self, x: Tensor<B, 4>, target_shape: [usize; 4]) -> Tensor<B, 4> {
        let downsample = self
            .downsample_input
            .as_ref()
            .expect("FOVNetwork encoder should have downsample module");
        let encoder = self
            .encoder
            .as_ref()
            .expect("FOVNetwork encoder missing ViT backbone");
        let projection = self
            .encoder_proj
            .as_ref()
            .expect("FOVNetwork encoder missing projection layer");

        let x = downsample.forward(x);
        let tokens = encoder.forward(x, None).x_norm_patchtokens;
        let dims: [usize; 3] = tokens.shape().dims();
        let batch = dims[0];
        let token_count = dims[1];
        let dim = dims[2];

        let tokens_flat = tokens
            .clone()
            .reshape([(batch * token_count) as i32, dim as i32]);
        let projected = projection.forward(tokens_flat);

        let projected = projected.reshape([
            batch as i32,
            token_count as i32,
            (self.num_features / 2) as i32,
        ]);
        let projected = projected.permute([0, 2, 1]);

        projected.reshape([
            target_shape[0] as i32,
            target_shape[1] as i32,
            target_shape[2] as i32,
            target_shape[3] as i32,
        ])
    }

    fn apply_blocks(&self, blocks: &[ConvActivation<B>], mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for block in blocks {
            x = block.forward(x);
        }
        x
    }
}
