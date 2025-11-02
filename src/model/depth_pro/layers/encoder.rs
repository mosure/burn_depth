use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
    },
    prelude::*,
};

use crate::model::{depth_pro::layers::vit::ViTConfig, dino::DinoVisionTransformer};

struct PatchSplit<B: Backend> {
    tensor: Tensor<B, 4>,
    steps: usize,
}

impl<B: Backend> PatchSplit<B> {
    fn new(tensor: Tensor<B, 4>, steps: usize) -> Self {
        Self { tensor, steps }
    }
}

#[derive(Module, Debug)]
struct ProjectUpsampleBlock<B: Backend> {
    projection: Conv2d<B>,
    upsample: Vec<ConvTranspose2d<B>>,
}

impl<B: Backend> ProjectUpsampleBlock<B> {
    fn new(
        device: &B::Device,
        dim_in: usize,
        dim_out: usize,
        upsample_layers: usize,
        dim_int: Option<usize>,
    ) -> Self {
        let intermediate = dim_int.unwrap_or(dim_out);
        let projection = Conv2dConfig::new([dim_in, intermediate], [1, 1])
            .with_bias(false)
            .init(device);

        let mut upsample = Vec::with_capacity(upsample_layers);
        for layer in 0..upsample_layers {
            let in_channels = if layer == 0 { intermediate } else { dim_out };
            upsample.push(
                ConvTranspose2dConfig::new([in_channels, dim_out], [2, 2])
                    .with_stride([2, 2])
                    .with_bias(false)
                    .init(device),
            );
        }

        Self {
            projection,
            upsample,
        }
    }

    fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        x = self.projection.forward(x);
        for layer in &self.upsample {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
pub struct DepthProEncoder<B: Backend> {
    dims_encoder: Vec<usize>,
    patch_encoder: DinoVisionTransformer<B>,
    image_encoder: DinoVisionTransformer<B>,
    hook_block_ids: Vec<usize>,
    decoder_features: usize,
    out_size: usize,
    img_size: usize,
    patch_window_size: usize,
    upsample_latent0: ProjectUpsampleBlock<B>,
    upsample_latent1: ProjectUpsampleBlock<B>,
    upsample0: ProjectUpsampleBlock<B>,
    upsample1: ProjectUpsampleBlock<B>,
    upsample2: ProjectUpsampleBlock<B>,
    upsample_lowres: ConvTranspose2d<B>,
    fuse_lowres: Conv2d<B>,
    half_downsample: Interpolate2d,
    quarter_downsample: Interpolate2d,
}

impl<B: Backend> DepthProEncoder<B> {
    pub fn new(
        device: &B::Device,
        dims_encoder: Vec<usize>,
        patch_encoder: DinoVisionTransformer<B>,
        patch_config: &ViTConfig,
        image_encoder: DinoVisionTransformer<B>,
        hook_block_ids: Vec<usize>,
        decoder_features: usize,
    ) -> Self {
        let out_size = patch_config.grid_size();
        let patch_window_size = patch_config.img_size;
        let img_size = patch_window_size * 4;

        let upsample_block = |dim_in, dim_out, upsample_layers, dim_int: Option<usize>| {
            ProjectUpsampleBlock::new(device, dim_in, dim_out, upsample_layers, dim_int)
        };

        let upsample_latent0 = upsample_block(
            patch_config.embed_dim,
            decoder_features,
            3,
            Some(dims_encoder[0]),
        );
        let upsample_latent1 = upsample_block(patch_config.embed_dim, dims_encoder[0], 2, None);
        let upsample0 = upsample_block(patch_config.embed_dim, dims_encoder[1], 1, None);
        let upsample1 = upsample_block(patch_config.embed_dim, dims_encoder[2], 1, None);
        let upsample2 = upsample_block(patch_config.embed_dim, dims_encoder[3], 1, None);

        let upsample_lowres = ConvTranspose2dConfig::new(
            [image_encoder.embedding_dimension(), dims_encoder[3]],
            [2, 2],
        )
        .with_stride([2, 2])
        .init(device);

        let fuse_lowres = Conv2dConfig::new([dims_encoder[3] * 2, dims_encoder[3]], [1, 1])
            .with_bias(true)
            .init(device);

        let half_downsample = Interpolate2dConfig::new()
            .with_scale_factor(Some([0.5, 0.5]))
            .with_mode(InterpolateMode::Linear)
            .init();
        let quarter_downsample = Interpolate2dConfig::new()
            .with_scale_factor(Some([0.25, 0.25]))
            .with_mode(InterpolateMode::Linear)
            .init();

        Self {
            dims_encoder,
            patch_encoder,
            image_encoder,
            hook_block_ids,
            decoder_features,
            out_size,
            img_size,
            patch_window_size,
            upsample_latent0,
            upsample_latent1,
            upsample0,
            upsample1,
            upsample2,
            upsample_lowres,
            fuse_lowres,
            half_downsample,
            quarter_downsample,
        }
    }

    pub fn img_size(&self) -> usize {
        self.img_size
    }

    fn split(&self, x: Tensor<B, 4>, overlap_ratio: f32) -> PatchSplit<B> {
        let dims: [usize; 4] = x.shape().dims();
        let batch = dims[0];
        let channels = dims[1];
        let image_size = dims[3];
        let patch_size = self.patch_window_size;
        let patch_stride = (patch_size as f32 * (1.0 - overlap_ratio)) as usize;

        let steps = if patch_size >= image_size {
            1
        } else {
            (((image_size - patch_size) as f32) / patch_stride as f32).ceil() as usize + 1
        };

        let mut patches = Vec::with_capacity(steps * steps);
        for j in 0..steps {
            let j0 = j * patch_stride;
            let j1 = j0 + patch_size;
            for i in 0..steps {
                let i0 = i * patch_stride;
                let i1 = i0 + patch_size;
                let patch = x.clone().slice([0..batch, 0..channels, j0..j1, i0..i1]);
                patches.push(patch);
            }
        }

        let tensor = if patches.len() == 1 {
            patches.pop().unwrap()
        } else {
            Tensor::cat(patches, 0)
        };

        PatchSplit::new(tensor, steps)
    }

    fn merge(&self, x: Tensor<B, 4>, batch_size: usize, padding: usize) -> Tensor<B, 4> {
        let dims: [usize; 4] = x.shape().dims();
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        let steps = ((dims[0] / batch_size) as f64).sqrt().round() as usize;
        let mut idx = 0;
        let mut rows = Vec::with_capacity(steps);

        for j in 0..steps {
            let mut cols = Vec::with_capacity(steps);
            for i in 0..steps {
                let start = batch_size * idx;
                let end = batch_size * (idx + 1);
                idx += 1;

                let mut patch = x
                    .clone()
                    .slice([start..end, 0..channels, 0..height, 0..width]);

                if j != 0 {
                    let dims: [usize; 4] = patch.shape().dims();
                    patch = patch.slice([0..batch_size, 0..dims[1], padding..dims[2], 0..dims[3]]);
                }
                if i != 0 {
                    let dims: [usize; 4] = patch.shape().dims();
                    patch = patch.slice([0..batch_size, 0..dims[1], 0..dims[2], padding..dims[3]]);
                }
                if j != steps - 1 {
                    let dims: [usize; 4] = patch.shape().dims();
                    patch = patch.slice([
                        0..batch_size,
                        0..dims[1],
                        0..(dims[2] - padding),
                        0..dims[3],
                    ]);
                }
                if i != steps - 1 {
                    let dims: [usize; 4] = patch.shape().dims();
                    patch = patch.slice([
                        0..batch_size,
                        0..dims[1],
                        0..dims[2],
                        0..(dims[3] - padding),
                    ]);
                }

                cols.push(patch);
            }
            rows.push(Tensor::cat(cols, 3));
        }

        Tensor::cat(rows, 2)
    }

    fn reshape_feature(
        &self,
        embeddings: Tensor<B, 3>,
        width: usize,
        height: usize,
        cls_token_offset: usize,
    ) -> Tensor<B, 4> {
        let dims: [usize; 3] = embeddings.shape().dims();
        let batch = dims[0];
        let tokens = dims[1];
        let dim = dims[2];

        let embeddings = if cls_token_offset > 0 {
            embeddings.slice([0..batch, cls_token_offset..tokens, 0..dim])
        } else {
            embeddings
        };

        embeddings
            .reshape([batch as i32, height as i32, width as i32, dim as i32])
            .permute([0, 3, 1, 2])
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let dims_root: [usize; 4] = x.shape().dims();
        let batch_size = dims_root[0];

        let x0 = x.clone();
        let x1 = self.half_downsample.forward(x.clone());
        let x2 = self.quarter_downsample.forward(x);

        let x0_split = self.split(x0, 0.25);
        let x1_split = self.split(x1, 0.5);
        let x2_split = PatchSplit::new(x2, 1);

        let x_pyramid_patches = Tensor::cat(
            vec![
                x0_split.tensor.clone(),
                x1_split.tensor.clone(),
                x2_split.tensor.clone(),
            ],
            0,
        );

        let (_patch_tokens, hook_tokens) = self
            .patch_encoder
            .forward_with_intermediate_tokens(x_pyramid_patches.clone(), &self.hook_block_ids);
        assert!(
            hook_tokens.len() >= 2,
            "DepthPro encoder expects at least two hook tokens"
        );
        let patch_output = self
            .patch_encoder
            .forward(x_pyramid_patches, None)
            .x_norm_patchtokens;

        let x_pyramid_encodings =
            self.reshape_feature(patch_output, self.out_size, self.out_size, 0);

        let len0 = x0_split.tensor.shape().dims::<4>()[0];
        let len1 = x1_split.tensor.shape().dims::<4>()[0];
        let len2 = x2_split.tensor.shape().dims::<4>()[0];
        let splits = x_pyramid_encodings
            .clone()
            .split_with_sizes(vec![len0, len1, len2], 0);

        let x0_encodings = splits[0].clone();
        let x1_encodings = splits[1].clone();
        let x2_encodings = splits[2].clone();

        let high_steps = x0_split.steps * x0_split.steps;
        let high_count = batch_size * high_steps;

        let latent0_encodings = {
            let enc = self.reshape_feature(hook_tokens[0].clone(), self.out_size, self.out_size, 1);
            let dims: [usize; 4] = enc.shape().dims();
            enc.slice([0..high_count, 0..dims[1], 0..dims[2], 0..dims[3]])
        };

        let latent1_encodings = {
            let enc = self.reshape_feature(hook_tokens[1].clone(), self.out_size, self.out_size, 1);
            let dims: [usize; 4] = enc.shape().dims();
            enc.slice([0..high_count, 0..dims[1], 0..dims[2], 0..dims[3]])
        };

        let x_latent0_features = self.merge(latent0_encodings, batch_size, 3);
        let x_latent1_features = self.merge(latent1_encodings, batch_size, 3);

        let x0_features = self.merge(x0_encodings, batch_size, 3);
        let x1_features = self.merge(x1_encodings, batch_size, 6);
        let x2_features = x2_encodings;

        let x_global_features = self.image_encoder.forward(x2_split.tensor.clone(), None);
        let mut x_global_features = self.reshape_feature(
            x_global_features.x_norm_patchtokens,
            self.out_size,
            self.out_size,
            0,
        );
        x_global_features = self.upsample_lowres.forward(x_global_features);
        let x2_features = self.upsample2.forward(x2_features);
        x_global_features = self
            .fuse_lowres
            .forward(Tensor::cat(vec![x2_features.clone(), x_global_features], 1));

        let x_latent0_features = self.upsample_latent0.forward(x_latent0_features);
        let x_latent1_features = self.upsample_latent1.forward(x_latent1_features);
        let x0_features = self.upsample0.forward(x0_features);
        let x1_features = self.upsample1.forward(x1_features);

        vec![
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]
    }
}
