use burn::tensor::activation::relu;
use burn::{
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        norm::BatchNorm,
        norm::BatchNormConfig,
    },
    prelude::*,
};

#[derive(Module, Debug)]
struct ProjectionConv<B: Backend> {
    conv: Option<Conv2d<B>>,
}

impl<B: Backend> ProjectionConv<B> {
    fn identity() -> Self {
        Self { conv: None }
    }

    fn new(
        device: &B::Device,
        channels: [usize; 2],
        kernel_size: [usize; 2],
        padding: usize,
        bias: bool,
    ) -> Self {
        let conv = Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .with_bias(bias)
            .init(device);
        Self { conv: Some(conv) }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if let Some(conv) = &self.conv {
            conv.forward(x)
        } else {
            x
        }
    }
}

#[derive(Module, Debug)]
struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: Option<BatchNorm<B>>,
    conv2: Conv2d<B>,
    bn2: Option<BatchNorm<B>>,
}

impl<B: Backend> ResidualBlock<B> {
    fn new(device: &B::Device, num_features: usize, batch_norm: bool) -> Self {
        let conv = |channels_in, channels_out| {
            Conv2dConfig::new([channels_in, channels_out], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(!batch_norm)
                .init(device)
        };

        let bn = |channels: usize| batch_norm.then(|| BatchNormConfig::new(channels).init(device));

        Self {
            conv1: conv(num_features, num_features),
            bn1: bn(num_features),
            conv2: conv(num_features, num_features),
            bn2: bn(num_features),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = relu(x);
        out = self.conv1.forward(out);
        if let Some(bn1) = &self.bn1 {
            out = bn1.forward(out);
        }
        out = relu(out);
        out = self.conv2.forward(out);
        if let Some(bn2) = &self.bn2 {
            out = bn2.forward(out);
        }
        out
    }
}

#[derive(Module, Debug)]
pub struct FeatureFusionBlock2d<B: Backend> {
    resnet1: ResidualBlock<B>,
    resnet2: ResidualBlock<B>,
    deconv: Option<ConvTranspose2d<B>>,
    out_conv: Conv2d<B>,
}

impl<B: Backend> FeatureFusionBlock2d<B> {
    pub fn new(device: &B::Device, num_features: usize, deconv: bool, batch_norm: bool) -> Self {
        let transposed = deconv.then(|| {
            ConvTranspose2dConfig::new([num_features, num_features], [2, 2])
                .with_stride([2, 2])
                .with_bias(false)
                .init(device)
        });

        let out_conv = Conv2dConfig::new([num_features, num_features], [1, 1])
            .with_bias(true)
            .init(device);

        Self {
            resnet1: ResidualBlock::new(device, num_features, batch_norm),
            resnet2: ResidualBlock::new(device, num_features, batch_norm),
            deconv: transposed,
            out_conv,
        }
    }

    pub fn forward(&self, x0: Tensor<B, 4>, x1: Option<Tensor<B, 4>>) -> Tensor<B, 4> {
        let mut x = x0;

        if let Some(ref residual) = x1 {
            let processed = self.resnet1.forward(residual.clone());
            x = x + processed;
        }

        x = self.resnet2.forward(x);

        if let Some(deconv) = &self.deconv {
            x = deconv.forward(x);
        }

        self.out_conv.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct MultiresConvDecoder<B: Backend> {
    dims_encoder: Vec<usize>,
    pub dim_decoder: usize,
    convs: Vec<ProjectionConv<B>>,
    fusions: Vec<FeatureFusionBlock2d<B>>,
}

impl<B: Backend> MultiresConvDecoder<B> {
    pub fn new(device: &B::Device, dims_encoder: Vec<usize>, dim_decoder: usize) -> Self {
        let mut convs = Vec::with_capacity(dims_encoder.len());

        if dims_encoder[0] != dim_decoder {
            convs.push(ProjectionConv::new(
                device,
                [dims_encoder[0], dim_decoder],
                [1, 1],
                0,
                false,
            ));
        } else {
            convs.push(ProjectionConv::identity());
        }

        for dim in dims_encoder.iter().skip(1) {
            convs.push(ProjectionConv::new(
                device,
                [*dim, dim_decoder],
                [3, 3],
                1,
                false,
            ));
        }

        let mut fusions = Vec::with_capacity(dims_encoder.len());
        for index in 0..dims_encoder.len() {
            fusions.push(FeatureFusionBlock2d::new(
                device,
                dim_decoder,
                index != 0,
                false,
            ));
        }

        Self {
            dims_encoder,
            dim_decoder,
            convs,
            fusions,
        }
    }

    pub fn forward(&self, encodings: &[Tensor<B, 4>]) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let num_levels = encodings.len();
        if num_levels != self.dims_encoder.len() {
            panic!(
                "Got encoder output levels = {num_levels}, expected {}.",
                self.dims_encoder.len()
            );
        }

        let mut features = self.convs[num_levels - 1].forward(encodings[num_levels - 1].clone());
        let lowres_features = features.clone();
        features = self.fusions[num_levels - 1].forward(features, None);

        for level in (0..num_levels - 1).rev() {
            let projected = self.convs[level].forward(encodings[level].clone());
            features = self.fusions[level].forward(features, Some(projected));
        }

        (features, lowres_features)
    }
}
