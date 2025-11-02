use burn::{
    module::Param,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        interpolate::InterpolateMode,
    },
    prelude::*,
};

pub mod layers {
    pub mod decoder;
    pub mod encoder;
    pub mod fov;
    pub mod vit;
}

use burn::tensor::activation::relu;
use layers::{
    decoder::MultiresConvDecoder,
    encoder::DepthProEncoder,
    fov::FOVNetwork,
    vit::{DINOV2_L16_384, create_vit},
};

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
        let x = relu(self.conv0.forward(x));
        let x = self.deconv.forward(x);
        let x = relu(self.conv1.forward(x));
        let x = self.conv_out.forward(x);
        relu(x)
    }
}

#[derive(Module, Debug)]
pub struct DepthPro<B: Backend> {
    encoder: DepthProEncoder<B>,
    decoder: MultiresConvDecoder<B>,
    head: DepthHead<B>,
    fov: Option<FOVNetwork<B>>,
}

pub struct DepthProInference<B: Backend> {
    pub depth: Tensor<B, 3>,
    pub focallength_px: Tensor<B, 1>,
}

impl<B: Backend> DepthPro<B> {
    pub fn new(device: &B::Device, config: DepthProConfig) -> Self {
        let (patch_encoder, patch_config) = create_vit(device, &config.patch_encoder_preset);
        let (image_encoder, _) = create_vit(device, &config.image_encoder_preset);

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
        );

        let mut decoder_dims = vec![config.decoder_features];
        decoder_dims.extend(patch_config.encoder_feature_dims.clone());

        let decoder = MultiresConvDecoder::new(device, decoder_dims, config.decoder_features);

        let head = DepthHead::new(device, config.decoder_features, (32, 1));

        let fov = config
            .use_fov_head
            .then(|| FOVNetwork::new(device, config.decoder_features, fov_encoder));

        Self {
            encoder,
            decoder,
            head,
            fov,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Option<Tensor<B, 1>>) {
        let encodings = self.encoder.forward(x.clone());
        let (features, lowres_features) = self.decoder.forward(&encodings);
        let canonical_inverse_depth = self.head.forward(features);

        let fov = self.fov.as_ref().map(|fov| {
            let fov_tensor = fov.forward(x, lowres_features);
            let dims_fov: [usize; 4] = fov_tensor.shape().dims();
            let batch = dims_fov[0];
            fov_tensor.reshape([batch as i32])
        });

        (canonical_inverse_depth, fov)
    }

    pub fn img_size(&self) -> usize {
        self.encoder.img_size()
    }

    pub fn infer(
        &self,
        mut x: Tensor<B, 4>,
        f_px: Option<Tensor<B, 1>>,
        interpolation_mode: InterpolateMode,
    ) -> DepthProInference<B> {
        let dims: [usize; 4] = x.shape().dims();
        let batch = dims[0];
        let height = dims[2];
        let width = dims[3];
        let resize_needed = self.encoder.img_size() != height || self.encoder.img_size() != width;

        if resize_needed {
            let resize = burn::nn::interpolate::Interpolate2dConfig::new()
                .with_output_size(Some([self.encoder.img_size(), self.encoder.img_size()]))
                .with_mode(interpolation_mode.clone())
                .init();
            x = resize.forward(x);
        }

        let (canonical_inverse_depth, fov_deg) = self.forward(x.clone());

        let mut focal_px = if let Some(f_px) = f_px {
            f_px.reshape([batch as i32, 1])
        } else {
            let fov_deg = fov_deg.expect("FOV head required for focal length");
            let radians = fov_deg.clone() * (core::f32::consts::PI / 180.0);
            let denom = (radians.clone() * 0.5).tan();
            let width_tensor = fov_deg.ones_like() * (width as f32 * 0.5);
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
            let resize_back = burn::nn::interpolate::Interpolate2dConfig::new()
                .with_output_size(Some([height, width]))
                .with_mode(interpolation_mode)
                .init();
            inverse_depth = resize_back.forward(inverse_depth);
        }

        let depth = inverse_depth.clamp(1e-4, 1e4).recip().squeeze_dim(1);

        DepthProInference {
            depth,
            focallength_px: focal_px.reshape([batch as i32]),
        }
    }
}
