use std::sync::{Arc, Mutex};

use burn::prelude::*;
use burn_depth::{
    inference::infer_from_rgb,
    model::{depth_anything3::DepthAnything3, prepare_depth_anything3_image, PreparedModelImage},
};
use image::{imageops, RgbImage};

pub mod platform;

pub async fn process_frame<B: Backend>(
    frame: RgbImage,
    model: Arc<Mutex<DepthAnything3<B>>>,
    device: B::Device,
    patch_size: usize,
    preferred_resolution: Option<usize>,
) -> Tensor<B, 3> {
    let patch_size = patch_size.max(1);
    let (frame, width, height) = prepare_input_frame(frame, patch_size, preferred_resolution);
    let pixels = frame.into_raw();

    let inference = {
        let guard = model.lock().expect("depth model poisoned");
        infer_from_rgb(&*guard, &pixels, width, height, &device).expect("failed to run inference")
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
        vec![normalized.clone(), normalized.clone(), normalized.clone()],
        2,
    );

    let alpha = Tensor::<B, 3>::ones([height, width, 1], &device);

    Tensor::<B, 3>::cat(vec![rgb, alpha], 2).reshape([height as i32, width as i32, 4])
}

fn prepare_input_frame(
    mut frame: RgbImage,
    patch_size: usize,
    preferred_resolution: Option<usize>,
) -> (RgbImage, usize, usize) {
    if let Some(mut target) = preferred_resolution {
        target = target.max(patch_size).max(1);
        let PreparedModelImage {
            rgb, width, height, ..
        } = prepare_depth_anything3_image(&frame, target)
            .expect("failed to prepare Depth Anything input frame");
        return (rgb, width, height);
    }

    let mut width = frame.width() as usize;
    let mut height = frame.height() as usize;
    let alignment = patch_size * 4;
    let align_down = |value: usize| -> usize {
        if value < patch_size {
            value
        } else if value >= alignment {
            let rem = value % alignment;
            if rem == 0 {
                value
            } else {
                value - rem
            }
        } else {
            let rem = value % patch_size;
            if rem == 0 {
                value
            } else {
                value - rem
            }
        }
    };

    let crop_width = align_down(width).max(1);
    let crop_height = align_down(height).max(1);

    if crop_width != width || crop_height != height {
        let offset_x = (width - crop_width) / 2;
        let offset_y = (height - crop_height) / 2;
        frame = imageops::crop_imm(
            &mut frame,
            offset_x as u32,
            offset_y as u32,
            crop_width as u32,
            crop_height as u32,
        )
        .to_image();
        width = crop_width;
        height = crop_height;
    }

    (frame, width, height)
}
