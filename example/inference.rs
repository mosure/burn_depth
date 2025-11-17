#![recursion_limit = "256"]

use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::prelude::*;
use burn_depth::{
    InferenceBackend,
    inference::{DepthPrediction, infer_from_rgb},
    model::{AnyDepthModel, DepthModelKind, ImageCropRegion, PreparedModelImage},
};
use clap::{Parser, ValueEnum};
use image::{GenericImageView, RgbImage};

#[derive(Parser, Debug)]
#[command(author, version, about = "Run depth inference using Burn checkpoints.")]
struct Args {
    #[arg(long, value_enum, default_value_t = ModelArg::DepthPro)]
    model: ModelArg,

    #[arg(long, value_name = "PATH")]
    checkpoint: Option<PathBuf>,

    #[arg(long, value_name = "PATH", default_value = "assets/image/test.jpg")]
    image: PathBuf,

    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelArg {
    DepthPro,
    DepthAnything3,
}

impl From<ModelArg> for DepthModelKind {
    fn from(value: ModelArg) -> Self {
        match value {
            ModelArg::DepthPro => DepthModelKind::DepthPro,
            ModelArg::DepthAnything3 => DepthModelKind::DepthAnything3,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let device = <InferenceBackend as Backend>::Device::default();
    let kind: DepthModelKind = args.model.into();

    let checkpoint = args
        .checkpoint
        .unwrap_or_else(|| PathBuf::from(kind.default_checkpoint()));
    if !checkpoint.exists() {
        return Err(format!(
            "Checkpoint `{}` not found. Provide --checkpoint or run the appropriate importer first.",
            checkpoint.display()
        )
        .into());
    }

    let model = AnyDepthModel::load(kind, &device, &checkpoint)?;

    let image_path = args.image;
    let image = image::open(&image_path)
        .map_err(|err| format!("Failed to load image `{}`: {err}", image_path.display()))?;
    let (orig_width, orig_height) = image.dimensions();
    let base_rgb = image.to_rgb8();
    let PreparedModelImage {
        width,
        height,
        rgb,
        crop: crop_region,
    } = model.prepare_input_image(&base_rgb)?;

    let result =
        infer_from_rgb::<InferenceBackend, _>(&model, rgb.as_raw(), width, height, &device)
            .map_err(|err| format!("Failed to run inference: {err}"))?;

    let restore_dims = if width as u32 != orig_width || height as u32 != orig_height {
        Some((orig_width as usize, orig_height as usize))
    } else if crop_region.is_some() {
        Some((orig_width as usize, orig_height as usize))
    } else {
        None
    };

    let depth_path = args
        .output
        .unwrap_or_else(|| image_path.with_file_name("depth.png"));
    save_depth_map(&result, &depth_path, crop_region, restore_dims)?;
    log_intrinsics(&result);
    println!(
        "Model `{}` wrote normalized depth map to {}",
        kind.as_str(),
        depth_path.display()
    );
    Ok(())
}

fn save_depth_map<B: Backend>(
    prediction: &DepthPrediction<B>,
    output_path: &Path,
    crop: Option<ImageCropRegion>,
    target_dims: Option<(usize, usize)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let depth_data = prediction.depth.clone().into_data().convert::<f32>();
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

    let mut values = depth_data
        .to_vec::<f32>()
        .map_err(|err| format!("Failed to read depth tensor values: {err:?}"))?;
    let mut save_width = width;
    let mut save_height = height;

    if let Some(region) = crop {
        values = crop_depth_field(&values, save_width, save_height, region)?;
        save_width = region.width;
        save_height = region.height;
    }

    if let Some((target_width, target_height)) = target_dims {
        if target_width != save_width || target_height != save_height {
            values = resize_depth_field(
                &values,
                save_width,
                save_height,
                target_width,
                target_height,
            );
            save_width = target_width;
            save_height = target_height;
        }
    }
    let (mut min_depth, mut max_depth) = (f32::INFINITY, f32::NEG_INFINITY);
    for &value in &values {
        if value.is_finite() {
            min_depth = min_depth.min(value);
            max_depth = max_depth.max(value);
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

    let width_u32 = u32::try_from(save_width)
        .map_err(|_| format!("Depth width {save_width} exceeds valid range"))?;
    let height_u32 = u32::try_from(save_height)
        .map_err(|_| format!("Depth height {save_height} exceeds valid range"))?;
    let depth_image = image::GrayImage::from_vec(width_u32, height_u32, pixels)
        .ok_or_else(|| format!("Depth tensor size mismatch {shape:?}"))?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    depth_image.save(output_path)?;
    Ok(())
}

fn resize_depth_field(
    values: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    if src_width == dst_width && src_height == dst_height {
        return values.to_vec();
    }
    let mut output = vec![0.0f32; dst_width * dst_height];
    let scale_x = if dst_width > 1 {
        src_width as f32 / dst_width as f32
    } else {
        0.0
    };
    let scale_y = if dst_height > 1 {
        src_height as f32 / dst_height as f32
    } else {
        0.0
    };

    for y in 0..dst_height {
        let src_y = if dst_height > 1 {
            (y as f32 + 0.5) * scale_y - 0.5
        } else {
            0.0
        };
        for x in 0..dst_width {
            let src_x = if dst_width > 1 {
                (x as f32 + 0.5) * scale_x - 0.5
            } else {
                0.0
            };
            output[y * dst_width + x] =
                sample_depth_bilinear(values, src_width, src_height, src_x, src_y);
        }
    }

    output
}

fn sample_depth_bilinear(values: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let clamp = |value: f32, max: usize| -> usize {
        if max == 0 {
            0
        } else {
            value.clamp(0.0, max as f32) as usize
        }
    };

    let x0 = clamp(x.floor(), width - 1);
    let y0 = clamp(y.floor(), height - 1);
    let x1 = clamp(x0 as f32 + 1.0, width - 1);
    let y1 = clamp(y0 as f32 + 1.0, height - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let index = |px: usize, py: usize| -> f32 { values[py * width + px] };

    let top = index(x0, y0) * (1.0 - fx) + index(x1, y0) * fx;
    let bottom = index(x0, y1) * (1.0 - fx) + index(x1, y1) * fx;

    top * (1.0 - fy) + bottom * fy
}

fn crop_depth_field(
    values: &[f32],
    width: usize,
    height: usize,
    region: ImageCropRegion,
) -> Result<Vec<f32>, String> {
    if region.x + region.width > width || region.y + region.height > height {
        return Err(format!(
            "Crop region {:?} exceeds depth tensor bounds {width}x{height}",
            region
        ));
    }
    let mut output = Vec::with_capacity(region.width * region.height);
    for row in 0..region.height {
        let src_y = region.y + row;
        let start = src_y * width + region.x;
        output.extend_from_slice(&values[start..start + region.width]);
    }
    Ok(output)
}

fn log_intrinsics<B: Backend>(prediction: &DepthPrediction<B>) {
    if let Some(focal) = prediction.focallength_px.clone() {
        let focal_values = focal
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap_or_default();
        println!("Focal length (px): {:?}", focal_values);
    } else {
        println!("Focal length (px): not provided by this model");
    }

    if let Some(fovy) = prediction.fovy_rad.clone() {
        let fovy_values = fovy
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap_or_default();
        println!("Vertical FOV (rad): {:?}", fovy_values);
    } else {
        println!("Vertical FOV (rad): not provided by this model");
    }
}
