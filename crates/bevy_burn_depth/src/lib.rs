use std::sync::{Arc, Mutex};

use burn::prelude::*;
use burn_depth::{
    inference::infer_from_rgb,
    model::depth_pro::DepthPro,
};
use image::RgbImage;

pub mod platform;

pub async fn process_frame<B: Backend>(
    frame: RgbImage,
    model: Arc<Mutex<DepthPro<B>>>,
    device: B::Device,
) -> Tensor<B, 3> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    let pixels = frame.into_raw();

    let inference = {
        let guard = model.lock().expect("depth model poisoned");
        infer_from_rgb(&*guard, &pixels, width, height, &device)
            .expect("failed to run inference")
    };

    let depth_map: Tensor<B, 2> = inference.depth.squeeze_dim(0);
    let dims: [usize; 2] = depth_map.shape().dims();
    let height = dims[0];
    let width = dims[1];

    let min_depth = depth_map
        .clone()
        .min()
        .into_scalar_async()
        .await
        .elem::<f32>();
    let max_depth = depth_map
        .clone()
        .max()
        .into_scalar_async()
        .await
        .elem::<f32>();
    let range = (max_depth - min_depth).max(f32::EPSILON);

    let normalized = depth_map
        .sub_scalar(min_depth)
        .div_scalar(range)
        .clamp(0.0, 1.0)
        .reshape([height as i32, width as i32, 1]);

    let rgb = Tensor::<B, 3>::cat(
        vec![
            normalized.clone(),
            normalized.clone(),
            normalized.clone(),
        ],
        2,
    );

    let alpha = Tensor::<B, 3>::ones([height, width, 1], &device);

    Tensor::<B, 3>::cat(vec![rgb, alpha], 2).reshape([height as i32, width as i32, 4])
}
