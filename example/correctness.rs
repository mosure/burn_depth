#![recursion_limit = "256"]

use std::{f32::consts::PI, path::Path};

use burn::{
    backend::Cuda,
    module::Module,
    nn::interpolate::InterpolateMode,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use burn_depth_pro::model::depth_pro::{DepthPro, DepthProConfig};
use image::GenericImageView;
use safetensors::tensor::SafeTensors;

type CorrectnessBackend = Cuda<f32>;

fn load_torch_reference(
    path: &Path,
) -> Result<(Vec<f32>, Vec<usize>, f32), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let depth_view = tensors
        .tensor("metric_depth")
        .map_err(|_| "missing `metric_depth` tensor in reference file")?;
    let depth_shape = depth_view.shape();
    if depth_shape.len() != 3 || depth_shape[2] != 1 {
        return Err(format!(
            "expected torch depth shape [H, W, 1], got {:?}",
            depth_shape
        )
        .into());
    }
    let depth = depth_view
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();

    let fovy_view = tensors
        .tensor("fovy")
        .map_err(|_| "missing `fovy` tensor in reference file")?;
    if fovy_view.shape() != [1] {
        return Err(format!("expected fovy shape [1], got {:?}", fovy_view.shape()).into());
    }
    let fovy = {
        let bytes = fovy_view.data();
        f32::from_le_bytes(bytes[..4].try_into().unwrap())
    };

    Ok((depth, depth_shape.to_vec(), fovy))
}

fn compute_burn_depth(
    image_path: &Path,
) -> Result<(Vec<f32>, usize, usize, f32), Box<dyn std::error::Error>> {
    let device = <CorrectnessBackend as Backend>::Device::default();

    let checkpoint = Path::new("assets/model/depth_pro.mpk");
    let model = DepthPro::<CorrectnessBackend>::new(&device, DepthProConfig::default());
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(checkpoint, &recorder, &device)
        .map_err(|err| format!("failed to load checkpoint: {err}"))?;

    let image = image::open(image_path)?;
    let (width, height) = image.dimensions();
    let image = image.to_rgb8();

    let mut data = Vec::with_capacity((3 * width * height) as usize);
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                data.push(image.get_pixel(x, y)[c] as f32 / 255.0);
            }
        }
    }

    let input: Tensor<CorrectnessBackend, 4> =
        Tensor::<CorrectnessBackend, 1>::from_floats(data.as_slice(), &device).reshape([
            1,
            3,
            height as usize,
            width as usize,
        ]);

    let output = model.infer(input, None, InterpolateMode::Linear);
    let depth_metric = output
        .depth
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to fetch burn depth: {err:?}"))?;
    let depth: Vec<f32> = depth_metric
        .into_iter()
        .map(|value| value)
        .collect();

    let f_px = output
        .focallength_px
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to fetch burn focal length: {err:?}"))?[0];

    let fovy = 2.0 * (0.5 * height as f32 / f_px).atan() * 180.0 / PI;

    Ok((depth, height as usize, width as usize, fovy))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = Path::new("assets/image/test.jpg");
    let reference_path = Path::new("assets/image/test.safetensors");

    if !reference_path.exists() {
        return Err(format!(
            "Torch reference `{}` missing. Run `python tool/correctness.py` first.",
            reference_path.display()
        )
        .into());
    }

    let (torch_depth, torch_shape, torch_fovy) = load_torch_reference(reference_path)?;
    let torch_height = torch_shape[0];
    let torch_width = torch_shape[1];

    let (burn_depth, burn_height, burn_width, burn_fovy) = compute_burn_depth(image_path)?;

    if burn_height != torch_height || burn_width != torch_width {
        return Err(format!(
            "shape mismatch: torch {torch_height}x{torch_width}, burn {burn_height}x{burn_width}"
        )
        .into());
    }

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut max_rel = 0.0f32;

    for (burn, reference) in burn_depth.iter().zip(torch_depth.iter()) {
        let diff = burn - reference;
        let abs = diff.abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;

        let rel = if reference.abs() > 1e-6 {
            abs / reference.abs()
        } else {
            0.0
        };
        max_rel = max_rel.max(rel);
    }

    let numel = burn_depth.len() as f32;
    let mean_abs = sum_abs / numel;

    let fovy_diff = (burn_fovy - torch_fovy).abs();

    println!("Pixel mean abs diff: {mean_abs:.6}");
    println!("Pixel max abs diff:  {max_abs:.6}");
    println!("Pixel max rel diff:  {max_rel:.6}");
    println!("FOVy difference:     {fovy_diff:.6} deg");

    const MAX_ABS_THRESHOLD: f32 = 5e-3; // ~1.5 mm
    const MEAN_ABS_THRESHOLD: f32 = 1e-3;
    const MAX_REL_THRESHOLD: f32 = 5e-3;
    const FOVY_THRESHOLD: f32 = 1e-3;

    if max_abs > MAX_ABS_THRESHOLD
        || mean_abs > MEAN_ABS_THRESHOLD
        || max_rel > MAX_REL_THRESHOLD
        || fovy_diff > FOVY_THRESHOLD
    {
        return Err("Burn output deviates from Torch reference beyond tolerance.".into());
    }

    println!("Burn output matches Torch reference within tolerance.");
    Ok(())
}
