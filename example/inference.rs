#![recursion_limit = "256"]

use std::{fs, path::Path};

#[allow(unused_imports)]
use burn::{
    backend::Cuda,
    module::Module,
    nn::interpolate::InterpolateMode,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use burn_depth_pro::{
    inference::infer_from_rgb,
    model::depth_pro::{DepthPro, DepthProConfig},
};
use image::GenericImageView;

type InferenceBackend = Cuda<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = <InferenceBackend as Backend>::Device::default();

    let checkpoint_path = Path::new("assets/model/depth_pro.mpk");
    if !checkpoint_path.exists() {
        return Err(format!(
            "Checkpoint `{}` not found. Run `cargo run --bin import --features import` first.",
            checkpoint_path.display()
        )
        .into());
    }

    let model = DepthPro::<InferenceBackend>::new(&device, DepthProConfig::default());
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(checkpoint_path, &recorder, &device)
        .map_err(|err| format!("Failed to load checkpoint: {err}"))?;

    let image_path = Path::new("assets/image/test.jpg");
    let image = image::open(image_path)
        .map_err(|err| format!("Failed to load image `{}`: {err}", image_path.display()))?;

    let (orig_width, orig_height) = image.dimensions();
    let rgb = image.to_rgb8();
    let width = orig_width as usize;
    let height = orig_height as usize;

    let result = infer_from_rgb::<InferenceBackend>(
        &model,
        rgb.as_raw(),
        width,
        height,
        &device,
        None,
        InterpolateMode::Linear,
    )
    .map_err(|err| format!("Failed to run inference: {err}"))?;

    let depth_data = result.depth.clone().into_data().convert::<f32>();
    let shape = depth_data.shape.clone();
    if shape.len() != 3 {
        return Err(format!("Expected depth tensor with 3 dimensions, got {shape:?}.").into());
    }
    let batch = shape[0];
    let height = shape[1];
    let width = shape[2];
    if batch != 1 {
        return Err(format!("Example expects batch size of 1, got {batch}.").into());
    }

    let values = depth_data
        .to_vec::<f32>()
        .map_err(|err| format!("Failed to read depth tensor values: {err:?}"))?;
    let (mut min_depth, mut max_depth) = (f32::INFINITY, f32::NEG_INFINITY);
    for &value in &values {
        if value.is_finite() {
            if value < min_depth {
                min_depth = value;
            }
            if value > max_depth {
                max_depth = value;
            }
        }
    }
    if !min_depth.is_finite() || !max_depth.is_finite() {
        min_depth = 0.0;
        max_depth = 1.0;
    }
    let range = (max_depth - min_depth).max(f32::EPSILON);
    let pixels: Vec<u8> = values
        .into_iter()
        .map(|value| {
            let normalized = if value.is_finite() {
                ((value - min_depth) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (normalized * 255.0).round().clamp(0.0, 255.0) as u8
        })
        .collect();

    let width_u32 = u32::try_from(width)
        .map_err(|_| format!("Depth width {width} exceeds supported output range"))?;
    let height_u32 = u32::try_from(height)
        .map_err(|_| format!("Depth height {height} exceeds supported output range"))?;
    let depth_image = image::GrayImage::from_vec(width_u32, height_u32, pixels)
        .ok_or_else(|| format!("Depth tensor size mismatch {shape:?}"))?;

    let output_path = image_path.with_file_name("test_depth.png");
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    depth_image.save(&output_path)?;

    let focal_data = result.focallength_px.clone().into_data().convert::<f32>();
    let focal_values = focal_data
        .to_vec::<f32>()
        .map_err(|err| format!("Failed to read focal length tensor: {err:?}"))?;
    println!("depth shape: {:?}", result.depth.shape());
    println!("focal length (px): {:?}", focal_values);
    println!("Saved normalized depth map to {}", output_path.display());

    Ok(())
}
