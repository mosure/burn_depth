use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::activation::relu,
};
use burn_dino::layers::{
    block::{Block, BlockConfig},
    layer_scale::LayerScaleConfig,
};

#[derive(Config, Debug)]
pub struct CameraEncoderConfig {
    pub dim_out: usize,
    pub dim_in: usize,
    pub trunk_depth: usize,
    pub target_dim: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    #[config(default = "0.01")]
    pub init_values: f32,
}

impl Default for CameraEncoderConfig {
    fn default() -> Self {
        Self {
            dim_out: 1024,
            dim_in: 9,
            trunk_depth: 4,
            target_dim: 9,
            num_heads: 16,
            mlp_ratio: 4,
            init_values: 0.01,
        }
    }
}

#[derive(Config, Debug)]
pub struct CameraDecoderConfig {
    pub dim_in: usize,
}

impl Default for CameraDecoderConfig {
    fn default() -> Self {
        Self { dim_in: 1536 }
    }
}

#[derive(Module, Debug)]
pub struct CameraEncoder<B: Backend> {
    pose_branch: PoseBranch<B>,
    trunk: Vec<Block<B>>,
    token_norm: LayerNorm<B>,
    trunk_norm: LayerNorm<B>,
    target_dim: usize,
}

impl<B: Backend> CameraEncoder<B> {
    pub fn new(device: &B::Device, config: CameraEncoderConfig) -> Self {
        let pose_branch = PoseBranch::new(device, config.dim_in, config.dim_out);
        let mut trunk = Vec::with_capacity(config.trunk_depth);
        let mut block_config = BlockConfig {
            attn: burn_dino::layers::attention::AttentionConfig {
                dim: config.dim_out,
                num_heads: config.num_heads,
                ..Default::default()
            },
            layer_scale: LayerScaleConfig {
                dim: config.dim_out,
            }
            .into(),
            mlp_ratio: config.mlp_ratio as f32,
        };
        block_config.attn.qkv_bias = true;
        block_config.attn.quiet_softmax = false;
        for _ in 0..config.trunk_depth {
            trunk.push(block_config.init(device));
        }
        Self {
            pose_branch,
            trunk,
            token_norm: LayerNormConfig::new(config.dim_out).init(device),
            trunk_norm: LayerNormConfig::new(config.dim_out).init(device),
            target_dim: config.target_dim,
        }
    }

    pub fn forward(
        &self,
        extrinsics: Tensor<B, 4>,
        intrinsics: Tensor<B, 4>,
        image_height: usize,
        image_width: usize,
    ) -> Tensor<B, 2> {
        let pose_encoding = extri_intri_to_pose_encoding(
            extrinsics,
            intrinsics,
            image_height,
            image_width,
            self.target_dim,
        );
        let mut tokens = self.pose_branch.forward(pose_encoding);
        tokens = self.token_norm.forward(tokens);
        for block in &self.trunk {
            tokens = block.forward(tokens.clone(), None, None);
        }
        let tokens = self.trunk_norm.forward(tokens);
        tokens.mean_dim(1).squeeze_dim::<2>(1)
    }
}

#[derive(Module, Debug)]
pub struct CameraDecoder<B: Backend> {
    backbone_1: Linear<B>,
    backbone_2: Linear<B>,
    fc_t: Linear<B>,
    fc_qvec: Linear<B>,
    fc_fov: Linear<B>,
}

impl<B: Backend> CameraDecoder<B> {
    pub fn new(device: &B::Device, config: CameraDecoderConfig) -> Self {
        Self {
            backbone_1: LinearConfig::new(config.dim_in, config.dim_in)
                .with_bias(true)
                .init(device),
            backbone_2: LinearConfig::new(config.dim_in, config.dim_in)
                .with_bias(true)
                .init(device),
            fc_t: LinearConfig::new(config.dim_in, 3)
                .with_bias(true)
                .init(device),
            fc_qvec: LinearConfig::new(config.dim_in, 4)
                .with_bias(true)
                .init(device),
            fc_fov: LinearConfig::new(config.dim_in, 2)
                .with_bias(true)
                .init(device),
        }
    }

    pub fn forward(
        &self,
        features: Tensor<B, 3>,
        camera_encoding: Option<Tensor<B, 3>>,
        image_height: usize,
        image_width: usize,
    ) -> CameraPrediction<B> {
        let dims = features.shape().dims::<3>();
        let batch = dims[0];
        let views = dims[1];
        let channels = dims[2];
        let flattened = features.reshape([batch as i32 * views as i32, channels as i32]);
        let hidden = relu(
            self.backbone_2
                .forward(relu(self.backbone_1.forward(flattened))),
        );

        let trans = self.fc_t.forward(hidden.clone());
        let quat = if let Some(encoding) = camera_encoding.as_ref() {
            encoding
                .clone()
                .reshape([
                    batch as i32 * views as i32,
                    encoding.shape().dims::<3>()[2] as i32,
                ])
                .slice([0..batch as i32 * views as i32, 3..7])
        } else {
            self.fc_qvec.forward(hidden.clone())
        };

        let fov = if let Some(encoding) = camera_encoding.as_ref() {
            encoding
                .clone()
                .reshape([
                    batch as i32 * views as i32,
                    encoding.shape().dims::<3>()[2] as i32,
                ])
                .slice([
                    0..batch as i32 * views as i32,
                    (encoding.shape().dims::<3>()[2] - 2) as i32
                        ..encoding.shape().dims::<3>()[2] as i32,
                ])
        } else {
            relu(self.fc_fov.forward(hidden.clone()))
        };

        let pose_encoding =
            Tensor::cat(vec![trans, quat, fov], 1).reshape([batch as i32, views as i32, 9]);
        let (extrinsics, intrinsics) =
            pose_encoding_to_extri_intri(pose_encoding.clone(), image_height, image_width);
        CameraPrediction {
            pose_encoding,
            extrinsics,
            intrinsics,
        }
    }
}

pub struct CameraPrediction<B: Backend> {
    pub pose_encoding: Tensor<B, 3>,
    pub extrinsics: Tensor<B, 4>,
    pub intrinsics: Tensor<B, 4>,
}

#[derive(Module, Debug)]
struct PoseBranch<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Gelu,
    drop: Dropout,
}

impl<B: Backend> PoseBranch<B> {
    fn new(device: &B::Device, dim_in: usize, dim_out: usize) -> Self {
        Self {
            fc1: LinearConfig::new(dim_in, dim_out / 2)
                .with_bias(true)
                .init(device),
            fc2: LinearConfig::new(dim_out / 2, dim_out)
                .with_bias(true)
                .init(device),
            activation: Gelu::new(),
            drop: DropoutConfig::new(0.0).init(),
        }
    }

    fn forward(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.activation.forward(self.fc1.forward(tensor));
        self.fc2.forward(self.drop.forward(x))
    }
}

fn extri_intri_to_pose_encoding<B: Backend>(
    extrinsics: Tensor<B, 4>,
    intrinsics: Tensor<B, 4>,
    image_height: usize,
    image_width: usize,
    target_dim: usize,
) -> Tensor<B, 3> {
    let device = extrinsics.device();
    let dims = extrinsics.shape().dims::<4>();
    let batch = dims[0];
    let views = dims[1];
    let total = (batch * views) as i32;

    let w2c = extrinsics.reshape([total, 3, 4]);
    let rotation = w2c.clone().slice([0..total, 0..3, 0..3]);
    let translation = w2c.slice([0..total, 0..3, 3..4]).reshape([total, 3, 1]);

    let c2w_rotation = rotation.clone().permute([0, 2, 1]);
    let c2w_translation = -c2w_rotation.clone().matmul(translation).squeeze_dim::<2>(2);
    let quaternion = matrix_to_quaternion(c2w_rotation.clone(), &device);

    let intr = intrinsics.reshape([total, 3, 3]);
    let fx = intr.clone().slice([0..total, 0..1, 0..1]).reshape([total]);
    let fy = intr.slice([0..total, 1..2, 1..2]).reshape([total]);

    let width_half = scalar_tensor(total, image_width as f32 / 2.0, &device);
    let height_half = scalar_tensor(total, image_height as f32 / 2.0, &device);
    let fov_w = approx_atan_positive(width_half.clone() / fx, &device) * 2.0;
    let fov_h = approx_atan_positive(height_half.clone() / fy, &device) * 2.0;

    let translation_flat = c2w_translation.clone();
    let fov_tensor = Tensor::cat(
        vec![
            fov_h.clone().reshape([total, 1]),
            fov_w.clone().reshape([total, 1]),
        ],
        1,
    );

    Tensor::cat(vec![translation_flat, quaternion.clone(), fov_tensor], 1).reshape([
        batch as i32,
        views as i32,
        target_dim as i32,
    ])
}

fn pose_encoding_to_extri_intri<B: Backend>(
    pose: Tensor<B, 3>,
    image_height: usize,
    image_width: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let device = pose.device();
    let dims = pose.shape().dims::<3>();
    let batch = dims[0];
    let views = dims[1];
    let total = (batch * views) as i32;
    let flat = pose.clone().reshape([total, 9]);

    let translation = flat.clone().slice([0..total, 0..3]).reshape([total, 3, 1]);
    let quaternion = flat.clone().slice([0..total, 3..7]);
    let fov = flat.slice([0..total, 7..9]);

    let rotation = quaternion_to_matrix(quaternion, &device);
    let rotation_t = rotation.clone().permute([0, 2, 1]);
    let translation_w2c =
        (-rotation_t.clone().matmul(translation).squeeze_dim::<2>(2)).unsqueeze_dim::<3>(2);
    let extrinsics = Tensor::cat(vec![rotation_t, translation_w2c], 2).reshape([
        batch as i32,
        views as i32,
        3,
        4,
    ]);

    let fov_h = fov.clone().slice([0..total, 0..1]).reshape([total]);
    let fov_w = fov.slice([0..total, 1..2]).reshape([total]);
    let half = scalar_tensor(total, 0.5, &device);
    let tan_half_h = (fov_h.clone() * half.clone()).sin() / (fov_h.clone() * half.clone()).cos();
    let tan_half_w = (fov_w.clone() * half.clone()).sin() / (fov_w.clone() * half.clone()).cos();
    let height_half = scalar_tensor(total, image_height as f32 / 2.0, &device);
    let width_half = scalar_tensor(total, image_width as f32 / 2.0, &device);
    let fy = height_half.clone() / tan_half_h;
    let fx = width_half.clone() / tan_half_w;
    let zeros = scalar_tensor(total, 0.0, &device);
    let ones = scalar_tensor(total, 1.0, &device);

    let row0 = Tensor::cat(
        vec![
            fx.clone().reshape([total, 1]),
            zeros.clone().reshape([total, 1]),
            width_half.clone().reshape([total, 1]),
        ],
        1,
    );
    let row1 = Tensor::cat(
        vec![
            zeros.clone().reshape([total, 1]),
            fy.clone().reshape([total, 1]),
            height_half.clone().reshape([total, 1]),
        ],
        1,
    );
    let row2 = Tensor::cat(
        vec![
            zeros.clone().reshape([total, 1]),
            zeros.clone().reshape([total, 1]),
            ones.clone().reshape([total, 1]),
        ],
        1,
    );

    let intrinsics = Tensor::cat(
        vec![
            row0.clone().unsqueeze_dim::<3>(1),
            row1.clone().unsqueeze_dim::<3>(1),
            row2.clone().unsqueeze_dim::<3>(1),
        ],
        1,
    )
    .reshape([batch as i32, views as i32, 3, 3]);

    (extrinsics, intrinsics)
}

fn scalar_tensor<B: Backend>(len: i32, value: f32, device: &B::Device) -> Tensor<B, 1> {
    Tensor::<B, 1>::ones([len], device) * value
}

fn quaternion_to_matrix<B: Backend>(quat: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 3> {
    let total = quat.shape().dims::<2>()[0] as i32;
    let x = quat.clone().slice([0..total, 0..1]).reshape([total]);
    let y = quat.clone().slice([0..total, 1..2]).reshape([total]);
    let z = quat.clone().slice([0..total, 2..3]).reshape([total]);
    let w = quat.slice([0..total, 3..4]).reshape([total]);

    let xx = x.clone() * x.clone();
    let yy = y.clone() * y.clone();
    let zz = z.clone() * z.clone();
    let xy = x.clone() * y.clone();
    let xz = x.clone() * z.clone();
    let yz = y.clone() * z.clone();
    let wx = w.clone() * x.clone();
    let wy = w.clone() * y.clone();
    let wz = w.clone() * z.clone();

    let ones = scalar_tensor(total, 1.0, device);
    let two = ones.clone() * 2.0;

    let row0 = Tensor::cat(
        vec![
            (ones.clone() - two.clone() * (yy.clone() + zz.clone())).reshape([total, 1]),
            (two.clone() * (xy.clone() - wz.clone())).reshape([total, 1]),
            (two.clone() * (xz.clone() + wy.clone())).reshape([total, 1]),
        ],
        1,
    );
    let row1 = Tensor::cat(
        vec![
            (two.clone() * (xy.clone() + wz.clone())).reshape([total, 1]),
            (ones.clone() - two.clone() * (xx.clone() + zz.clone())).reshape([total, 1]),
            (two.clone() * (yz.clone() - wx.clone())).reshape([total, 1]),
        ],
        1,
    );
    let row2 = Tensor::cat(
        vec![
            (two.clone() * (xz.clone() - wy.clone())).reshape([total, 1]),
            (two.clone() * (yz.clone() + wx.clone())).reshape([total, 1]),
            (ones.clone() - two.clone() * (xx.clone() + yy.clone())).reshape([total, 1]),
        ],
        1,
    );

    Tensor::cat(
        vec![
            row0.clone().unsqueeze_dim::<3>(1),
            row1.clone().unsqueeze_dim::<3>(1),
            row2.clone().unsqueeze_dim::<3>(1),
        ],
        1,
    )
    .reshape([total, 3, 3])
}

fn matrix_to_quaternion<B: Backend>(rotation: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 2> {
    let total = rotation.shape().dims::<3>()[0] as i32;
    let m00 = rotation
        .clone()
        .slice([0..total, 0..1, 0..1])
        .reshape([total]);
    let m01 = rotation
        .clone()
        .slice([0..total, 0..1, 1..2])
        .reshape([total]);
    let m02 = rotation
        .clone()
        .slice([0..total, 0..1, 2..3])
        .reshape([total]);
    let m10 = rotation
        .clone()
        .slice([0..total, 1..2, 0..1])
        .reshape([total]);
    let m11 = rotation
        .clone()
        .slice([0..total, 1..2, 1..2])
        .reshape([total]);
    let m12 = rotation
        .clone()
        .slice([0..total, 1..2, 2..3])
        .reshape([total]);
    let m20 = rotation
        .clone()
        .slice([0..total, 2..3, 0..1])
        .reshape([total]);
    let m21 = rotation
        .clone()
        .slice([0..total, 2..3, 1..2])
        .reshape([total]);
    let m22 = rotation.slice([0..total, 2..3, 2..3]).reshape([total]);

    let ones = scalar_tensor(total, 1.0, device);
    let quarter = ones.clone() * 0.25;
    let eps = scalar_tensor(total, 1e-6, device);

    let trace = m00.clone() + m11.clone() + m22.clone();
    let s_trace = ((trace.clone() + ones.clone()).clamp_min(1e-6).sqrt() * 2.0).reshape([total]);
    let qw_trace = (quarter.clone() * s_trace.clone()).reshape([total, 1]);
    let qx_trace = ((m21.clone() - m12.clone()) / s_trace.clone()).reshape([total, 1]);
    let qy_trace = ((m02.clone() - m20.clone()) / s_trace.clone()).reshape([total, 1]);
    let qz_trace = ((m10.clone() - m01.clone()) / s_trace.clone()).reshape([total, 1]);
    let quat_trace = Tensor::stack(vec![qx_trace, qy_trace, qz_trace, qw_trace], 1);

    let s_x = ((ones.clone() + m00.clone() - m11.clone() - m22.clone())
        .clamp_min(1e-6)
        .sqrt()
        * 2.0)
        .reshape([total]);
    let qx_x = (quarter.clone() * s_x.clone()).reshape([total, 1]);
    let qw_x = ((m21.clone() - m12.clone()) / (s_x.clone() + eps.clone())).reshape([total, 1]);
    let qy_x = ((m01.clone() + m10.clone()) / (s_x.clone() + eps.clone())).reshape([total, 1]);
    let qz_x = ((m02.clone() + m20.clone()) / (s_x.clone() + eps.clone())).reshape([total, 1]);
    let quat_x = Tensor::stack(vec![qx_x, qy_x, qz_x, qw_x], 1);

    let s_y = ((ones.clone() + m11.clone() - m00.clone() - m22.clone())
        .clamp_min(1e-6)
        .sqrt()
        * 2.0)
        .reshape([total]);
    let qy_y = (quarter.clone() * s_y.clone()).reshape([total, 1]);
    let qw_y = ((m02.clone() - m20.clone()) / (s_y.clone() + eps.clone())).reshape([total, 1]);
    let qx_y = ((m01.clone() + m10.clone()) / (s_y.clone() + eps.clone())).reshape([total, 1]);
    let qz_y = ((m12.clone() + m21.clone()) / (s_y.clone() + eps.clone())).reshape([total, 1]);
    let quat_y = Tensor::stack(vec![qx_y, qy_y, qz_y, qw_y], 1);

    let s_z = ((ones.clone() + m22.clone() - m00.clone() - m11.clone())
        .clamp_min(1e-6)
        .sqrt()
        * 2.0)
        .reshape([total]);
    let qz_z = (quarter.clone() * s_z.clone()).reshape([total, 1]);
    let qw_z = ((m10 - m01) / (s_z.clone() + eps.clone())).reshape([total, 1]);
    let qx_z = ((m02.clone() + m20.clone()) / (s_z.clone() + eps.clone())).reshape([total, 1]);
    let qy_z = ((m12.clone() + m21.clone()) / (s_z.clone() + eps.clone())).reshape([total, 1]);
    let quat_z = Tensor::stack(vec![qx_z, qy_z, qz_z, qw_z], 1);

    let mask_trace = trace.clone().greater_elem(0.0).float();
    let cond_x =
        m00.clone().greater(m11.clone()).float() * m00.clone().greater(m22.clone()).float();
    let mask_x = (ones.clone() - mask_trace.clone()) * cond_x;
    let cond_y = m11.clone().greater(m22.clone()).float();
    let mask_y = (ones.clone() - mask_trace.clone() - mask_x.clone()) * cond_y;
    let mask_z = ones.clone() - mask_trace.clone() - mask_x.clone() - mask_y.clone();

    let mask_trace = mask_trace.reshape([total, 1]);
    let mask_x = mask_x.reshape([total, 1]);
    let mask_y = mask_y.reshape([total, 1]);
    let mask_z = mask_z.reshape([total, 1]);

    quat_trace * mask_trace + quat_x * mask_x + quat_y * mask_y + quat_z * mask_z
}

fn approx_atan_positive<B: Backend>(x: Tensor<B, 1>, device: &B::Device) -> Tensor<B, 1> {
    let dims = x.shape().dims::<1>()[0] as i32;
    let ones = scalar_tensor(dims, 1.0, device);
    let abs_x = x.clone();
    let coeff_a = scalar_tensor(dims, 0.2447, device);
    let coeff_b = scalar_tensor(dims, 0.0663, device);
    let pi_over_4 = scalar_tensor(dims, core::f32::consts::FRAC_PI_4, device);
    let pi_over_2 = scalar_tensor(dims, core::f32::consts::FRAC_PI_2, device);

    let approximation = |v: Tensor<B, 1>| {
        let term = coeff_a.clone() + coeff_b.clone() * v.clone();
        (pi_over_4.clone() * v.clone()) - v.clone() * (v.clone() - ones.clone()) * term
    };

    let small = approximation(abs_x.clone());
    let inv = ones.clone() / abs_x.clone().clamp_min(1e-6);
    let large = pi_over_2.clone() - approximation(inv);

    let mask_small = abs_x.clone().lower_equal_elem(1.0).float();
    let mask_large = ones.clone() - mask_small.clone();
    small * mask_small + large * mask_large
}
