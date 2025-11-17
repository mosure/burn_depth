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
    let extr_data = extrinsics.into_data().convert::<f32>();
    let intr_data = intrinsics.into_data().convert::<f32>();
    let extr_values = extr_data
        .to_vec::<f32>()
        .expect("extrinsics tensor conversion");
    let intr_values = intr_data
        .to_vec::<f32>()
        .expect("intrinsics tensor conversion");
    let shape = extr_data.shape.clone();
    let batch = shape[0];
    let views = shape[1];
    let rows = shape[2];
    let cols = shape[3];
    let mut pose = Vec::with_capacity(batch * views * target_dim);
    for b in 0..batch {
        for v in 0..views {
            let base = ((b * views + v) * rows * cols) as usize;
            let matrix = &extr_values[base..base + rows * cols];
            let mut w2c = [0.0f32; 16];
            copy_3x4_into_homogeneous(matrix, rows, cols, &mut w2c);
            let c2w = affine_inverse_host(&w2c);
            let translation = [c2w[3], c2w[7], c2w[11]];
            let rotation = [
                c2w[0], c2w[1], c2w[2], c2w[4], c2w[5], c2w[6], c2w[8], c2w[9], c2w[10],
            ];
            let quat = mat_to_quat_host(&rotation);

            let intr_base = ((b * views + v) * 9) as usize;
            let fx = intr_values[intr_base];
            let fy = intr_values[intr_base + 4];
            let fov_h = 2.0 * ((image_height as f32 / 2.0) / fy).atan();
            let fov_w = 2.0 * ((image_width as f32 / 2.0) / fx).atan();

            pose.extend_from_slice(&translation);
            pose.extend_from_slice(&quat);
            pose.push(fov_h);
            pose.push(fov_w);
        }
    }
    Tensor::<B, 1>::from_floats(pose.as_slice(), &device).reshape([
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
    let data = pose.into_data().convert::<f32>();
    let values = data.to_vec::<f32>().expect("pose tensor to vec");
    let shape = data.shape.clone();
    let batch = shape[0];
    let views = shape[1];
    let mut extrinsics = Vec::with_capacity(batch * views * 12);
    let mut intrinsics = Vec::with_capacity(batch * views * 9);
    for b in 0..batch {
        for v in 0..views {
            let base = ((b * views + v) * 9) as usize;
            let translation = [values[base], values[base + 1], values[base + 2]];
            let quat = [
                values[base + 3],
                values[base + 4],
                values[base + 5],
                values[base + 6],
            ];
            let fov_h = values[base + 7];
            let fov_w = values[base + 8];
            let rotation = quat_to_mat_host(quat);

            let mut c2w = [0.0f32; 12];
            c2w[0] = rotation[0];
            c2w[1] = rotation[1];
            c2w[2] = rotation[2];
            c2w[3] = translation[0];
            c2w[4] = rotation[3];
            c2w[5] = rotation[4];
            c2w[6] = rotation[5];
            c2w[7] = translation[1];
            c2w[8] = rotation[6];
            c2w[9] = rotation[7];
            c2w[10] = rotation[8];
            c2w[11] = translation[2];
            let w2c = invert_c2w_to_w2c(&c2w);
            extrinsics.extend_from_slice(&w2c);

            let fy = (image_height as f32 / 2.0) / (0.5 * fov_h).tan();
            let fx = (image_width as f32 / 2.0) / (0.5 * fov_w).tan();
            intrinsics.extend_from_slice(&[
                fx,
                0.0,
                image_width as f32 / 2.0,
                0.0,
                fy,
                image_height as f32 / 2.0,
                0.0,
                0.0,
                1.0,
            ]);
        }
    }

    let extr_tensor = Tensor::<B, 1>::from_floats(extrinsics.as_slice(), &device).reshape([
        batch as i32,
        views as i32,
        3,
        4,
    ]);

    let intr_tensor = Tensor::<B, 1>::from_floats(intrinsics.as_slice(), &device).reshape([
        batch as i32,
        views as i32,
        3,
        3,
    ]);

    (extr_tensor, intr_tensor)
}

fn affine_inverse_host(matrix: &[f32]) -> [f32; 16] {
    let r = [
        matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
        matrix[10],
    ];
    let t = [matrix[3], matrix[7], matrix[11]];
    let rt = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
    let tx = -rt[0] * t[0] - rt[1] * t[1] - rt[2] * t[2];
    let ty = -rt[3] * t[0] - rt[4] * t[1] - rt[5] * t[2];
    let tz = -rt[6] * t[0] - rt[7] * t[1] - rt[8] * t[2];
    [
        rt[0], rt[1], rt[2], tx, rt[3], rt[4], rt[5], ty, rt[6], rt[7], rt[8], tz, 0.0, 0.0, 0.0,
        1.0,
    ]
}

fn mat_to_quat_host(matrix: &[f32]) -> [f32; 4] {
    let trace = matrix[0] + matrix[4] + matrix[8];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (matrix[7] - matrix[5]) / s,
            (matrix[2] - matrix[6]) / s,
            (matrix[3] - matrix[1]) / s,
            0.25 * s,
        ]
    } else if matrix[0] > matrix[4] && matrix[0] > matrix[8] {
        let s = (1.0 + matrix[0] - matrix[4] - matrix[8]).sqrt() * 2.0;
        [
            0.25 * s,
            (matrix[1] + matrix[3]) / s,
            (matrix[2] + matrix[6]) / s,
            (matrix[7] - matrix[5]) / s,
        ]
    } else if matrix[4] > matrix[8] {
        let s = (1.0 + matrix[4] - matrix[0] - matrix[8]).sqrt() * 2.0;
        [
            (matrix[1] + matrix[3]) / s,
            0.25 * s,
            (matrix[5] + matrix[7]) / s,
            (matrix[2] - matrix[6]) / s,
        ]
    } else {
        let s = (1.0 + matrix[8] - matrix[0] - matrix[4]).sqrt() * 2.0;
        [
            (matrix[2] + matrix[6]) / s,
            (matrix[5] + matrix[7]) / s,
            0.25 * s,
            (matrix[3] - matrix[1]) / s,
        ]
    }
}

fn quat_to_mat_host(quat: [f32; 4]) -> [f32; 9] {
    let [x, y, z, w] = quat;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    [
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy + wz),
        2.0 * (xz - wy),
        2.0 * (xy - wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz + wx),
        2.0 * (xz + wy),
        2.0 * (yz - wx),
        1.0 - 2.0 * (xx + yy),
    ]
}

fn copy_3x4_into_homogeneous(src: &[f32], rows: usize, cols: usize, dst: &mut [f32; 16]) {
    dst.fill(0.0);
    for r in 0..rows.min(3) {
        for c in 0..4 {
            let value = if c < cols { src[r * cols + c] } else { 0.0 };
            dst[r * 4 + c] = value;
        }
    }
    dst[15] = 1.0;
}

fn invert_c2w_to_w2c(matrix: &[f32; 12]) -> [f32; 12] {
    let mut homo = [0.0f32; 16];
    copy_3x4_into_homogeneous(matrix, 3, 4, &mut homo);
    let inv = affine_inverse_host(&homo);
    [
        inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], inv[6], inv[7], inv[8], inv[9], inv[10],
        inv[11],
    ]
}
