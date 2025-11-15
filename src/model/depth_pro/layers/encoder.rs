use burn::{
    module::Ignored,
    nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    prelude::*,
};

use crate::model::depth_pro::{InterpolationMethod, layers::vit::ViTConfig, resize_bilinear_scale};
use burn_dino::model::dino::DinoVisionTransformer;

#[derive(Clone)]
struct PatchSplit<B: Backend> {
    tensor: Tensor<B, 4>,
    steps: usize,
    patch_size: usize,
    stride: usize,
}

impl<B: Backend> PatchSplit<B> {
    fn new(tensor: Tensor<B, 4>, steps: usize, patch_size: usize, stride: usize) -> Self {
        Self {
            tensor,
            steps,
            patch_size,
            stride,
        }
    }

    fn feature_padding(&self, feature_patch_size: usize) -> usize {
        if feature_patch_size == 0 || self.patch_size == 0 {
            return 0;
        }

        let denom = self.patch_size.max(1);
        let feature_stride = (self.stride * feature_patch_size + denom / 2) / denom;
        feature_patch_size
            .saturating_sub(feature_stride)
            .saturating_div(2)
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
    interpolation: Ignored<InterpolationMethod>,
}

pub struct EncoderDebug<B: Backend> {
    pub features: Vec<Tensor<B, 4>>,
    pub latent0: Tensor<B, 4>,
    pub latent1: Tensor<B, 4>,
    pub latent0_tokens: Tensor<B, 4>,
    pub latent1_tokens: Tensor<B, 4>,
    pub latent0_merge_input: Tensor<B, 4>,
    pub latent1_merge_input: Tensor<B, 4>,
    pub x0_tokens: Tensor<B, 4>,
    pub x1_tokens: Tensor<B, 4>,
    pub x2_tokens: Tensor<B, 4>,
    pub split_x0: Tensor<B, 4>,
    pub split_x1: Tensor<B, 4>,
    pub split_x2: Tensor<B, 4>,
    pub merged_x0: Tensor<B, 4>,
    pub merged_x1: Tensor<B, 4>,
    pub merged_x2: Tensor<B, 4>,
}

impl<B: Backend> DepthProEncoder<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        dims_encoder: Vec<usize>,
        patch_encoder: DinoVisionTransformer<B>,
        patch_config: &ViTConfig,
        image_encoder: DinoVisionTransformer<B>,
        image_embed_dim: usize,
        hook_block_ids: Vec<usize>,
        decoder_features: usize,
        interpolation: InterpolationMethod,
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

        let upsample_lowres =
            ConvTranspose2dConfig::new([image_embed_dim, dims_encoder[3]], [2, 2])
                .with_stride([2, 2])
                .init(device);

        let fuse_lowres = Conv2dConfig::new([dims_encoder[3] * 2, dims_encoder[3]], [1, 1])
            .with_bias(true)
            .init(device);

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
            interpolation: Ignored(interpolation),
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
        let mut patch_stride =
            ((patch_size as f32 * (1.0 - overlap_ratio)).floor() as usize).max(1);
        if patch_stride > patch_size {
            patch_stride = patch_size;
        }

        let steps = if patch_size >= image_size {
            1
        } else {
            1 + (image_size - patch_size).div_ceil(patch_stride)
        };

        let mut patches = Vec::with_capacity(steps * steps);
        for j in 0..steps {
            let j0 = j * patch_stride;
            let j1 = j0 + patch_size;
            for i in 0..steps {
                let i0 = i * patch_stride;
                let i1 = i0 + patch_size;

                patches.push(
                    x.clone()
                        .slice([0..batch, 0..channels, j0..j1, i0..i1])
                        .reshape([
                            (batch) as i32,
                            channels as i32,
                            patch_size as i32,
                            patch_size as i32,
                        ]),
                );
            }
        }

        let tensor = Tensor::cat(patches, 0);

        PatchSplit::new(tensor, steps, patch_size, patch_stride)
    }

    fn merge(&self, x: Tensor<B, 4>, batch_size: usize, padding: usize) -> Tensor<B, 4> {
        let dims: [usize; 4] = x.shape().dims();
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        let steps = ((dims[0] / batch_size) as f64).sqrt().round() as usize;
        if steps == 0 {
            return Tensor::zeros([batch_size as i32, channels as i32, 0, 0], &x.device());
        }

        let mut rows: Vec<Tensor<B, 4>> = Vec::with_capacity(steps);
        for j in 0..steps {
            let mut row_patches: Vec<Tensor<B, 4>> = Vec::with_capacity(steps);
            for i in 0..steps {
                let idx = j * steps + i;
                let start = batch_size * idx;
                let end = start + batch_size;

                let mut patch = x
                    .clone()
                    .slice([start..end, 0..channels, 0..height, 0..width]);

                let top_trim = if j == 0 { 0 } else { padding };
                let bottom_trim = if j == steps - 1 { 0 } else { padding };
                let left_trim = if i == 0 { 0 } else { padding };
                let right_trim = if i == steps - 1 { 0 } else { padding };

                let bottom_index = height - bottom_trim;
                let right_index = width - right_trim;

                if top_trim != 0 || bottom_trim != 0 || left_trim != 0 || right_trim != 0 {
                    patch = patch.slice([
                        0..batch_size,
                        0..channels,
                        top_trim..bottom_index,
                        left_trim..right_index,
                    ]);
                }

                row_patches.push(patch);
            }

            let row = Tensor::cat(row_patches, 3);
            rows.push(row);
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

    pub fn forward_with_debug(&self, x: Tensor<B, 4>) -> EncoderDebug<B> {
        let dims_root: [usize; 4] = x.shape().dims();
        let batch_size = dims_root[0];

        let x0 = x.clone();
        let x1 = resize_bilinear_scale(x.clone(), [0.5, 0.5], self.interpolation.0);
        let x2 = resize_bilinear_scale(x, [0.25, 0.25], self.interpolation.0);

        let x0_split = self.split(x0, 0.25);
        let x1_split = self.split(x1, 0.5);
        let x2_dims: [usize; 4] = x2.shape().dims();
        let x2_split = PatchSplit::new(x2, 1, x2_dims[2], x2_dims[2]);
        let split_x0_tensor = x0_split.tensor.clone();
        let split_x1_tensor = x1_split.tensor.clone();
        let split_x2_tensor = x2_split.tensor.clone();

        let x_pyramid_patches = Tensor::cat(
            vec![
                x0_split.tensor.clone(),
                x1_split.tensor.clone(),
                x2_split.tensor.clone(),
            ],
            0,
        );

        let (patch_output, hook_tokens) = self
            .patch_encoder
            .forward_with_intermediate_tokens(x_pyramid_patches.clone(), &self.hook_block_ids);
        assert!(
            hook_tokens.len() >= 2,
            "DepthPro encoder expects at least two hook tokens"
        );

        let x_pyramid_encodings = self.reshape_feature(
            patch_output.x_norm_patchtokens,
            self.out_size,
            self.out_size,
            0,
        );

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

        let latent0_merge_input =
            self.reshape_feature(hook_tokens[0].clone(), self.out_size, self.out_size, 1);
        let latent1_merge_input =
            self.reshape_feature(hook_tokens[1].clone(), self.out_size, self.out_size, 1);
        let latent0_encodings = {
            let dims: [usize; 4] = latent0_merge_input.shape().dims();
            latent0_merge_input
                .clone()
                .slice([0..high_count, 0..dims[1], 0..dims[2], 0..dims[3]])
        };
        let latent1_encodings = {
            let dims: [usize; 4] = latent1_merge_input.shape().dims();
            latent1_merge_input
                .clone()
                .slice([0..high_count, 0..dims[1], 0..dims[2], 0..dims[3]])
        };

        let high_padding = x0_split.feature_padding(self.out_size);
        let mid_padding = x1_split.feature_padding(self.out_size);

        let latent0_tokens_clone = latent0_encodings.clone();
        let latent1_tokens_clone = latent1_encodings.clone();

        let merged_latent0 = self.merge(latent0_encodings, batch_size, high_padding);
        let merged_latent1 = self.merge(latent1_encodings, batch_size, high_padding);

        let x0_tokens_clone = x0_encodings.clone();
        let x1_tokens_clone = x1_encodings.clone();
        let x2_tokens_clone = x2_encodings.clone();

        let merged_x0 = self.merge(x0_encodings, batch_size, high_padding);
        let merged_x1 = self.merge(x1_encodings, batch_size, mid_padding);
        let merged_x2 = x2_encodings;

        let x_global_features = self.image_encoder.forward(x2_split.tensor.clone(), None);
        let mut x_global_features = self.reshape_feature(
            x_global_features.x_norm_patchtokens,
            self.out_size,
            self.out_size,
            0,
        );
        x_global_features = self.upsample_lowres.forward(x_global_features);
        let upsampled_x2 = self.upsample2.forward(merged_x2.clone());
        x_global_features = self.fuse_lowres.forward(Tensor::cat(
            vec![upsampled_x2.clone(), x_global_features],
            1,
        ));

        let upsampled_latent0 = self.upsample_latent0.forward(merged_latent0.clone());
        let upsampled_latent1 = self.upsample_latent1.forward(merged_latent1.clone());
        let upsampled_x0 = self.upsample0.forward(merged_x0.clone());
        let upsampled_x1 = self.upsample1.forward(merged_x1.clone());

        let features = vec![
            upsampled_latent0.clone(),
            upsampled_latent1.clone(),
            upsampled_x0.clone(),
            upsampled_x1.clone(),
            x_global_features,
        ];

        EncoderDebug {
            features,
            latent0: merged_latent0,
            latent1: merged_latent1,
            latent0_tokens: latent0_tokens_clone,
            latent1_tokens: latent1_tokens_clone,
            latent0_merge_input,
            latent1_merge_input,
            x0_tokens: x0_tokens_clone,
            x1_tokens: x1_tokens_clone,
            x2_tokens: x2_tokens_clone,
            split_x0: split_x0_tensor,
            split_x1: split_x1_tensor,
            split_x2: split_x2_tensor,
            merged_x0,
            merged_x1,
            merged_x2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        self.forward_with_debug(x).features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::depth_pro::layers::vit::{DINOV2_L16_128, create_vit};

    type TestBackend = crate::InferenceBackend;

    fn build_encoder(
        device: &<TestBackend as Backend>::Device,
    ) -> (DepthProEncoder<TestBackend>, ViTConfig) {
        let (patch_encoder, patch_config) = create_vit::<TestBackend>(device, DINOV2_L16_128);
        let (image_encoder, image_config) = create_vit::<TestBackend>(device, DINOV2_L16_128);

        let encoder = DepthProEncoder::new(
            device,
            patch_config.encoder_feature_dims.clone(),
            patch_encoder,
            &patch_config,
            image_encoder,
            image_config.embed_dim,
            patch_config.encoder_feature_layer_ids.clone(),
            64,
            InterpolationMethod::Custom,
        );

        (encoder, patch_config)
    }

    fn make_ordered_input(
        device: &<TestBackend as Backend>::Device,
        channels: usize,
        size: usize,
    ) -> Tensor<TestBackend, 4> {
        let total = channels * size * size;
        let values: Vec<f32> = (0..total).map(|idx| idx as f32).collect();

        Tensor::<TestBackend, 1>::from_floats(values.as_slice(), device)
            .reshape([1, channels, size, size])
    }

    #[test]
    fn split_merge_roundtrip_without_overlap() {
        let device = <TestBackend as Backend>::Device::default();
        let (encoder, patch_config) = build_encoder(&device);
        let image_size = encoder.img_size();
        let input = make_ordered_input(&device, 3, image_size);
        let baseline = input.clone();

        let split = encoder.split(input.clone(), 0.0);
        assert_eq!(split.steps * split.steps, 16);
        let batch_size = baseline.shape().dims::<4>()[0];
        let padding = split.feature_padding(patch_config.grid_size());
        let merged = encoder.merge(split.tensor, batch_size, padding);

        assert!(
            merged.clone().all_close(baseline, Some(1e-5), Some(1e-5)),
            "split/merge without overlap should reconstruct the input"
        );
    }

    #[test]
    fn merge_overlapping_layout_matches_expected() {
        let device = <TestBackend as Backend>::Device::default();
        let (encoder, _config) = build_encoder(&device);

        let batch_size = 1;
        let channels = 2;
        let feature_size = 8;
        let steps = 5;
        let padding = 1;
        let patch_count = batch_size * steps * steps;
        let patch_volume = channels * feature_size * feature_size;

        let patch_values: Vec<f32> = (0..patch_count)
            .flat_map(|patch_idx| std::iter::repeat(patch_idx as f32).take(patch_volume))
            .collect();

        let patches = Tensor::<TestBackend, 1>::from_floats(patch_values.as_slice(), &device)
            .reshape([patch_count, channels, feature_size, feature_size]);
        let merged = encoder.merge(patches, batch_size, padding);
        let dims: [usize; 4] = merged.shape().dims();
        let output_height = dims[2];
        let output_width = dims[3];

        let mut expected = vec![-1f32; batch_size * channels * output_height * output_width];
        for batch in 0..batch_size {
            for j in 0..steps {
                for i in 0..steps {
                    let patch_idx = batch_size * (j * steps + i) + batch;
                    let top_trim = if j == 0 { 0 } else { padding };
                    let bottom_trim = if j == steps - 1 { 0 } else { padding };
                    let left_trim = if i == 0 { 0 } else { padding };
                    let right_trim = if i == steps - 1 { 0 } else { padding };

                    let slice_height = feature_size - top_trim - bottom_trim;
                    let slice_width = feature_size - left_trim - right_trim;
                    let base_y =
                        j * (feature_size - 2 * padding) + if j == 0 { 0 } else { padding };
                    let base_x =
                        i * (feature_size - 2 * padding) + if i == 0 { 0 } else { padding };

                    for channel in 0..channels {
                        for dy in 0..slice_height {
                            for dx in 0..slice_width {
                                let y = base_y + dy;
                                let x = base_x + dx;
                                let offset = ((((batch * channels) + channel) * output_height) + y)
                                    * output_width
                                    + x;
                                expected[offset] = patch_idx as f32;
                            }
                        }
                    }
                }
            }
        }

        let merged_vec = merged.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(merged_vec.len(), expected.len());
        for (observed, reference) in merged_vec.iter().zip(expected.iter()) {
            assert!(
                (*observed - *reference).abs() < 1e-5,
                "merge produced {observed} but expected {reference}"
            );
        }
    }
}
