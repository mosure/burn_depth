use burn::prelude::*;

use crate::model::{
    AnyDepthModel,
    depth_anything3::DepthAnything3,
    depth_pro::{DepthPro, DepthProInference},
};

#[derive(Debug, Clone)]
pub struct DepthPrediction<B: Backend> {
    pub depth: Tensor<B, 3>,
    pub focallength_px: Option<Tensor<B, 1>>,
    pub fovy_rad: Option<Tensor<B, 1>>,
}

impl<B: Backend> DepthPrediction<B> {
    pub fn has_intrinsics(&self) -> bool {
        self.focallength_px.is_some() || self.fovy_rad.is_some()
    }
}

pub trait DepthModel<B: Backend> {
    fn infer_depth(&self, input: Tensor<B, 4>) -> DepthPrediction<B>;
}

impl<B: Backend> From<DepthProInference<B>> for DepthPrediction<B> {
    fn from(value: DepthProInference<B>) -> Self {
        Self {
            depth: value.depth,
            focallength_px: Some(value.focallength_px),
            fovy_rad: Some(value.fovy_rad),
        }
    }
}

impl<B: Backend> DepthModel<B> for DepthPro<B> {
    fn infer_depth(&self, input: Tensor<B, 4>) -> DepthPrediction<B> {
        self.infer(input).into()
    }
}

impl<B: Backend> DepthModel<B> for DepthAnything3<B> {
    fn infer_depth(&self, input: Tensor<B, 4>) -> DepthPrediction<B> {
        let result = self.infer(input);
        DepthPrediction {
            depth: result.depth,
            focallength_px: None,
            fovy_rad: None,
        }
    }
}

impl<B: Backend> DepthModel<B> for AnyDepthModel<B> {
    fn infer_depth(&self, input: Tensor<B, 4>) -> DepthPrediction<B> {
        match self {
            AnyDepthModel::DepthPro(model) => model.infer_depth(input),
            AnyDepthModel::DepthAnything3(model) => model.infer_depth(input),
        }
    }
}

/// Converts packed RGB bytes into a normalized tensor suitable for `DepthPro::infer`.
///
/// The input slice must contain `width * height * 3` bytes in row-major order.
/// Each pixel is converted to floats in `[0, 1]`, then normalized with the ImageNet
/// mean / standard deviation expected by the DINO encoder. The output tensor is
/// channel-first (`NCHW`).
pub fn rgb_to_input_tensor<B: Backend>(
    rgb: &[u8],
    width: usize,
    height: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4>, String> {
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "image dimensions overflowed while preparing input".to_string())?;

    if rgb.len() != expected_len {
        return Err(format!(
            "expected {expected_len} RGB bytes for {width}x{height}, got {}",
            rgb.len()
        ));
    }

    let hw = width * height;
    let mut data = vec![0.0f32; 3 * hw];

    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    for (idx, pixel) in rgb.chunks_exact(3).enumerate() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        data[idx] = (r - MEAN[0]) / STD[0];
        data[hw + idx] = (g - MEAN[1]) / STD[1];
        data[2 * hw + idx] = (b - MEAN[2]) / STD[2];
    }

    Ok(
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([
            1,
            3,
            height as i32,
            width as i32,
        ]),
    )
}

/// Runs the DepthPro model directly from packed RGB bytes.
///
/// This helper combines [`rgb_to_input_tensor`] and [`DepthPro::infer`], making it
/// convenient to integrate inference in external applications without reimplementing
/// the preprocessing pipeline.
pub fn infer_from_rgb<B: Backend, M: DepthModel<B>>(
    model: &M,
    rgb: &[u8], // TODO: use an image type here
    width: usize,
    height: usize,
    device: &B::Device,
) -> Result<DepthPrediction<B>, String> {
    let input = rgb_to_input_tensor::<B>(rgb, width, height, device)?;
    Ok(model.infer_depth(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = crate::InferenceBackend;

    #[test]
    fn rgb_to_input_tensor_normalizes_channels() {
        let device = <TestBackend as Backend>::Device::default();
        let rgb = vec![
            0u8, 255, 128, //
            255, 0, 128,
        ];
        let tensor = rgb_to_input_tensor::<TestBackend>(&rgb, 1, 2, &device).unwrap();
        let data = tensor.into_data().convert::<f32>();
        assert_eq!(data.shape.as_slice(), &[1, 3, 2, 1]);
        let values = data.to_vec::<f32>().unwrap();

        // Expected values computed using f32 precision to align with PyTorch's normalization.
        let expected = [
            -2.1179039,
            2.2489083,
            2.4285715,
            -2.0357141,
            0.42649257,
            0.42649257,
        ];
        assert_eq!(values.len(), expected.len());
        for (value, expected) in values.iter().zip(expected.iter()) {
            assert!(
                (value - expected).abs() < 1e-2,
                "value {value} diverged from expected {expected}"
            );
        }
    }

    #[test]
    fn rgb_to_input_tensor_rejects_invalid_length() {
        let device = <TestBackend as Backend>::Device::default();
        let rgb = vec![0u8; 5];
        let result = rgb_to_input_tensor::<TestBackend>(&rgb, 1, 2, &device);
        assert!(result.is_err());
    }
}
