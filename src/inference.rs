use burn::prelude::*;

use crate::model::depth_pro::{DepthPro, DepthProInference};

/// Converts packed RGB bytes into a normalized tensor suitable for `DepthPro::infer`.
///
/// The input slice must contain `width * height * 3` bytes in row-major order.
/// The output tensor is channel-first (`NCHW`) with values scaled to `[-1, 1]`.
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

    for (idx, pixel) in rgb.chunks_exact(3).enumerate() {
        for channel in 0..3 {
            let value = pixel[channel] as f32 / 255.0;
            data[channel * hw + idx] = value * 2.0 - 1.0;
        }
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
pub fn infer_from_rgb<B: Backend>(
    model: &DepthPro<B>,
    rgb: &[u8],
    width: usize,
    height: usize,
    device: &B::Device,
    focal_length_px: Option<Tensor<B, 1>>,
) -> Result<DepthProInference<B>, String> {
    let input = rgb_to_input_tensor::<B>(rgb, width, height, device)?;
    Ok(model.infer(input, focal_length_px))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

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

        let expected = [-1.0f32, 1.0f32, 1.0f32, -1.0f32, 0.0039215689, 0.0039215689];
        assert_eq!(values.len(), expected.len());
        for (value, expected) in values.iter().zip(expected.iter()) {
            assert!((value - expected).abs() < 1e-6);
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
