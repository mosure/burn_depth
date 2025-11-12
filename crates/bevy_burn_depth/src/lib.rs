use std::sync::{Arc, Mutex};

use burn::prelude::*;
use burn_depth::{inference::rgb_to_input_tensor, model::depth_pro::DepthPro};
use image::{imageops::FilterType, DynamicImage, RgbImage};

pub mod platform;

fn resize_frame(input: RgbImage, target_size: usize) -> (Vec<u8>, usize, usize) {
    let resized = DynamicImage::ImageRgb8(input)
        .resize_exact(target_size as u32, target_size as u32, FilterType::Triangle)
        .to_rgb8();

    let width = resized.width() as usize;
    let height = resized.height() as usize;

    (resized.into_raw(), width, height)
}

fn normalize_depth_rgba<B: Backend>(
    depth: Tensor<B, 2>,
    min_value: f32,
    max_value: f32,
    device: &B::Device,
) -> Tensor<B, 3> {
    let dims: [usize; 2] = depth.shape().dims();
    let height = dims[0];
    let width = dims[1];

    let range = (max_value - min_value).max(1e-6);

    let normalized = depth
        .sub_scalar(min_value)
        .div_scalar(range)
        .clamp(0.0, 1.0)
        .reshape([height as i32, width as i32, 1]);

    let rgb = Tensor::<B, 3>::cat(vec![normalized.clone(), normalized.clone(), normalized], 2);
    let alpha = Tensor::<B, 3>::ones([height, width, 1], device);

    Tensor::<B, 3>::cat(vec![rgb, alpha], 2)
}

pub async fn process_frame<B: Backend>(
    frame: RgbImage,
    model: Arc<Mutex<DepthPro<B>>>,
    device: B::Device,
) -> Tensor<B, 3> {
    let target_size = {
        let guard = model.lock().expect("depth model poisoned");
        guard.img_size()
    };

    let (pixels, width, height) = resize_frame(frame, target_size);
    let input =
        rgb_to_input_tensor::<B>(&pixels, width, height, &device).expect("invalid frame data");

    let inference = {
        let guard = model.lock().expect("depth model poisoned");
        guard.infer(input)
    };

    let depth_map = inference.depth.squeeze_dim(0);

    let min_value = depth_map
        .clone()
        .min()
        .into_scalar_async()
        .await
        .elem::<f32>();
    let max_value = depth_map
        .clone()
        .max()
        .into_scalar_async()
        .await
        .elem::<f32>();

    normalize_depth_rgba(depth_map, min_value, max_value, &device)
}
