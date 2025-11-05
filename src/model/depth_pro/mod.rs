use burn::{
    module::{Ignored, Module, Param},
    nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    prelude::*,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, RecorderError},
};

pub mod layers {
    pub mod decoder;
    pub mod encoder;
    pub mod fov;
    pub mod vit;
}
mod interpolate;

use burn::tensor::activation::relu;
pub use interpolate::{
    InterpolationMethod, resize_bilinear_align_corners_false, resize_bilinear_scale,
};
use layers::{
    decoder::MultiresConvDecoder,
    encoder::{DepthProEncoder, EncoderDebug},
    fov::FOVNetwork,
    vit::{DINOV2_L16_384, create_vit},
};

pub type ForwardDecoderOutputs<B> = (
    Tensor<B, 4>,
    Tensor<B, 4>,
    Tensor<B, 4>,
    Vec<Tensor<B, 4>>,
    Option<Tensor<B, 1>>,
);

#[derive(Config, Debug)]
pub struct DepthProConfig {
    pub patch_encoder_preset: String,
    pub image_encoder_preset: String,
    pub decoder_features: usize,

    #[config(default = "None")]
    pub checkpoint_uri: Option<String>,

    #[config(default = "None")]
    pub fov_encoder_preset: Option<String>,

    #[config(default = "true")]
    pub use_fov_head: bool,

    #[config(default = "InterpolationMethod::Custom")]
    pub interpolation: InterpolationMethod,
}

impl Default for DepthProConfig {
    fn default() -> Self {
        Self {
            patch_encoder_preset: DINOV2_L16_384.into(),
            image_encoder_preset: DINOV2_L16_384.into(),
            decoder_features: 256,
            checkpoint_uri: None,
            fov_encoder_preset: Some(DINOV2_L16_384.into()),
            use_fov_head: true,
            interpolation: InterpolationMethod::Custom,
        }
    }
}

#[derive(Module, Debug)]
struct DepthHead<B: Backend> {
    conv0: Conv2d<B>,
    deconv: ConvTranspose2d<B>,
    conv1: Conv2d<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> DepthHead<B> {
    fn new(device: &B::Device, dim_decoder: usize, last_dims: (usize, usize)) -> Self {
        let conv0 = Conv2dConfig::new([dim_decoder, dim_decoder / 2], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let deconv = ConvTranspose2dConfig::new([dim_decoder / 2, dim_decoder / 2], [2, 2])
            .with_stride([2, 2])
            .with_bias(true)
            .init(device);
        let conv1 = Conv2dConfig::new([dim_decoder / 2, last_dims.0], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let mut conv_out = Conv2dConfig::new([last_dims.0, last_dims.1], [1, 1])
            .with_bias(true)
            .init(device);

        if let Some(bias) = conv_out.bias.as_ref() {
            let zeros = Tensor::<B, 1>::zeros(bias.val().shape(), device);
            conv_out.bias = Some(Param::from_tensor(zeros));
        }

        Self {
            conv0,
            deconv,
            conv1,
            conv_out,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv0.forward(x);
        let x = self.deconv.forward(x);
        let x = self.conv1.forward(x);
        let x = relu(x);
        let x = self.conv_out.forward(x);
        relu(x)
    }

    fn fix_conv_transpose_weights(&mut self) {
        maybe_fix_conv_transpose2d(&mut self.deconv);
    }
}

#[derive(Module, Debug)]
pub struct DepthPro<B: Backend> {
    encoder: DepthProEncoder<B>,
    decoder: MultiresConvDecoder<B>,
    head: DepthHead<B>,
    fov: Option<FOVNetwork<B>>,
    interpolation: Ignored<InterpolationMethod>,
}

pub struct DepthProInference<B: Backend> {
    pub depth: Tensor<B, 3>,
    pub focallength_px: Tensor<B, 1>,
    pub fovx_deg: Tensor<B, 1>,
    pub fovy_rad: Tensor<B, 1>,
}

pub struct HeadDebug<B: Backend> {
    pub conv0: Tensor<B, 4>,
    pub deconv: Tensor<B, 4>,
    pub conv1: Tensor<B, 4>,
    pub relu: Tensor<B, 4>,
    pub pre_out: Tensor<B, 4>,
    pub canonical: Tensor<B, 4>,
}

impl<B: Backend> DepthPro<B> {
    pub fn new(device: &B::Device, config: DepthProConfig) -> Self {
        let (patch_encoder, patch_config) = create_vit(device, &config.patch_encoder_preset);
        let (image_encoder, _) = create_vit(device, &config.image_encoder_preset);
        let interpolation = config.interpolation;

        let fov_encoder = config
            .fov_encoder_preset
            .as_ref()
            .filter(|_| config.use_fov_head)
            .map(|preset| create_vit(device, preset).0);

        let encoder = DepthProEncoder::new(
            device,
            patch_config.encoder_feature_dims.clone(),
            patch_encoder,
            &patch_config,
            image_encoder,
            patch_config.encoder_feature_layer_ids.clone(),
            config.decoder_features,
            interpolation,
        );

        let mut decoder_dims = vec![config.decoder_features];
        decoder_dims.extend(patch_config.encoder_feature_dims.clone());

        let decoder = MultiresConvDecoder::new(device, decoder_dims, config.decoder_features);

        let head = DepthHead::new(device, config.decoder_features, (32, 1));

        let fov = config
            .use_fov_head
            .then(|| FOVNetwork::new(device, config.decoder_features, fov_encoder, interpolation));

        Self {
            encoder,
            decoder,
            head,
            fov,
            interpolation: Ignored(interpolation),
        }
    }

    pub fn load(
        device: &B::Device,
        checkpoint_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, RecorderError> {
        Self::load_with_config(device, DepthProConfig::default(), checkpoint_path)
    }

    pub fn load_with_config(
        device: &B::Device,
        config: DepthProConfig,
        checkpoint_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, RecorderError> {
        let checkpoint_path = checkpoint_path.as_ref();
        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        Self::new(device, config).load_file(checkpoint_path, &recorder, device)
    }

    fn forward_internal(&self, x: Tensor<B, 4>) -> ForwardDecoderOutputs<B> {
        let encodings = self.encoder.forward(x.clone());
        let (features, lowres_features, fusion_outputs) =
            self.decoder.forward_with_debug(&encodings);
        let decoder_features = features.clone();
        let decoder_lowres_features = lowres_features.clone();
        // if cfg!(debug_assertions)
        //     && let Ok(stats) = features
        //         .clone()
        //         .into_data()
        //         .convert::<f32>()
        //         .to_vec::<f32>()
        // {
        //     let mut min_v = f32::INFINITY;
        //     let mut max_v = f32::NEG_INFINITY;
        //     let mut sum_v = 0.0f32;
        //     for value in &stats {
        //         min_v = min_v.min(*value);
        //         max_v = max_v.max(*value);
        //         sum_v += *value;
        //     }
        //     let mean_v = sum_v / stats.len() as f32;
        //     println!(
        //         "Burn decoder feature stats: min={min_v:.6}, max={max_v:.6}, mean={mean_v:.6}"
        //     );
        // }
        let canonical_inverse_depth = self.head.forward(features);

        let fov = self.fov.as_ref().map(|fov| {
            let fov_tensor = fov.forward(x, lowres_features);
            let dims_fov: [usize; 4] = fov_tensor.shape().dims();
            let batch = dims_fov[0];
            fov_tensor.reshape([batch as i32])
        });

        (
            canonical_inverse_depth,
            decoder_features,
            decoder_lowres_features,
            fusion_outputs,
            fov,
        )
    }

    pub fn fix_conv_transpose_weights(&mut self) {
        self.decoder.fix_conv_transpose_weights();
        self.head.fix_conv_transpose_weights();
        if let Some(fov) = self.fov.as_mut() {
            fov.fix_conv_transpose_weights();
        }
    }

    pub fn head_debug(&self, feature: Tensor<B, 4>) -> HeadDebug<B> {
        let conv0 = self.head.conv0.forward(feature);
        let deconv = self.head.deconv.forward(conv0.clone());
        let conv1 = self.head.conv1.forward(deconv.clone());
        let relu_out = relu(conv1.clone());
        let pre_out = self.head.conv_out.forward(relu_out.clone());
        let canonical = relu(pre_out.clone());

        HeadDebug {
            conv0,
            deconv,
            conv1,
            relu: relu_out,
            pre_out,
            canonical,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Option<Tensor<B, 1>>) {
        let (canonical, _, _, _, fov) = self.forward_internal(x);
        (canonical, fov)
    }

    pub fn forward_with_decoder(&self, x: Tensor<B, 4>) -> ForwardDecoderOutputs<B> {
        self.forward_internal(x)
    }

    pub fn decoder_from_features(
        &self,
        features: &[Tensor<B, 4>],
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        self.decoder.forward_with_debug(features)
    }

    pub fn img_size(&self) -> usize {
        self.encoder.img_size()
    }

    pub fn encoder_forward_debug(&self, x: Tensor<B, 4>) -> EncoderDebug<B> {
        self.encoder.forward_with_debug(x)
    }

    pub fn encoder_features(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        self.encoder.forward(x)
    }

    pub fn interpolation_method(&self) -> InterpolationMethod {
        self.interpolation.0
    }

    pub fn infer(&self, mut x: Tensor<B, 4>) -> DepthProInference<B> {
        let dims: [usize; 4] = x.shape().dims();
        let batch = dims[0];
        let height = dims[2];
        let width = dims[3];
        let resize_needed = self.encoder.img_size() != height || self.encoder.img_size() != width;

        if resize_needed {
            x = resize_bilinear_align_corners_false(
                x,
                [self.encoder.img_size(), self.encoder.img_size()],
                self.interpolation.0,
            );
        }

        let (canonical_inverse_depth, fov_deg) = self.forward(x.clone());

        let fovx_deg = fov_deg.expect("FOV head required for focal length");
        let fovx_rad = fovx_deg.clone() * (core::f32::consts::PI / 180.0);

        let mut focal_px = {
            let denom = (fovx_rad.clone() * 0.5).tan();
            let width_tensor = fovx_deg.ones_like() * (width as f32 * 0.5);
            (width_tensor / denom).reshape([batch as i32, 1])
        };

        let dims_focal: [usize; 2] = focal_px.shape().dims();
        if dims_focal[0] != batch {
            focal_px = focal_px.reshape([batch as i32, 1]);
        }

        let width_tensor = focal_px.ones_like() * (width as f32);
        let ratio = width_tensor.clone() / focal_px.clone();
        let ratio = ratio.reshape([batch as i32, 1, 1, 1]);
        let mut inverse_depth = canonical_inverse_depth * ratio;

        if resize_needed {
            inverse_depth = resize_bilinear_align_corners_false(
                inverse_depth,
                [height, width],
                self.interpolation.0,
            );
        }

        let depth = inverse_depth.clamp(1e-4, 1e4).recip().squeeze_dim(1);

        DepthProInference {
            depth,
            fovx_deg,
            fovy_rad: fovy_from_fovx_rad(fovx_rad, height, width),
            focallength_px: focal_px.reshape([batch as i32]),
        }
    }
}


/// fovy = 2 * atan( (H/W) * tan(fovx/2) )
/// uses raján atan approx on [0,1] with range reduction for |x|>1.
/// input: fovx_rad (radians), output: fovy_rad (radians).
pub fn fovy_from_fovx_rad<B: Backend, const D: usize>(
    fovx_rad: Tensor<B, D>,
    h: usize,
    w: usize,
) -> Tensor<B, D> {
    let k = 0.273;  // raján constant
    let pi_over_4 = core::f64::consts::FRAC_PI_4;
    let pi_over_2 = core::f64::consts::FRAC_PI_2;

    // aspect = H/W
    let aspect = (h as f64) / (w as f64);

    // t = (H/W) * tan(fovx/2)
    let t = fovx_rad.mul_scalar(0.5).tan().mul_scalar(aspect);

    // atan(t): range reduction + raján on [0,1]
    let s = t.clone().sign();
    let ax = t.abs();
    let use_inv = ax.clone().greater_elem(1.0).float();         // 1 where |x|>1 else 0
    let inv = ax.clone().recip();
    let xr = ax.mul(use_inv.clone().neg().add_scalar(1.0)) // (1-use_inv)*ax
        .add(inv.mul(use_inv.clone()));             // + use_inv*(1/ax)

    // atan(xr) ≈ xr * (π/4 + k*(1 - xr)), xr∈[0,1]
    let inner = xr.clone().neg().add_scalar(1.0).mul_scalar(k).add_scalar(pi_over_4);
    let atan_reduced = xr.mul(inner);

    // undo reduction: atan(ax) = atan_reduced + use_inv * (π/2 - 2*atan_reduced)
    let delta   = atan_reduced.clone().mul_scalar(2.0).neg().add_scalar(pi_over_2);
    let atan_ax = atan_reduced.add(delta.mul(use_inv));

    // restore sign and finish
    let atan_t = atan_ax.mul(s);
    atan_t.mul_scalar(2.0)
}


pub(crate) fn maybe_fix_conv_transpose2d<B: Backend>(conv: &mut ConvTranspose2d<B>) {
    let weight = conv.weight.val();
    let dims: [usize; 4] = weight.shape().dims();
    let expected_in = conv.channels[0];
    let expected_out = conv.channels[1] / conv.groups;

    if dims[0] == expected_in && dims[1] == expected_out {
        // Already in Burn layout.
        return;
    }

    if dims[0] == expected_out && dims[1] == expected_in {
        let permuted = weight.swap_dims(0, 1);
        conv.weight = Param::from_tensor(permuted);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use layers::vit::DINOV2_L16_128;

    type TestBackend = crate::InferenceBackend;

    fn small_config(interpolation: InterpolationMethod) -> DepthProConfig {
        DepthProConfig {
            patch_encoder_preset: DINOV2_L16_128.into(),
            image_encoder_preset: DINOV2_L16_128.into(),
            fov_encoder_preset: Some(DINOV2_L16_128.into()),
            decoder_features: 64,
            interpolation,
            ..DepthProConfig::default()
        }
    }

    #[test]
    fn interpolation_method_matches_configuration() {
        let device = <TestBackend as Backend>::Device::default();
        let custom_model =
            DepthPro::<TestBackend>::new(&device, small_config(InterpolationMethod::Custom));
        let burn_model =
            DepthPro::<TestBackend>::new(&device, small_config(InterpolationMethod::Burn));

        assert_eq!(
            custom_model.interpolation_method(),
            InterpolationMethod::Custom
        );
        assert_eq!(burn_model.interpolation_method(), InterpolationMethod::Burn);
    }
}
