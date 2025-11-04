#![recursion_limit = "256"]

use std::{convert::TryInto, f32::consts::PI, path::Path};

use burn::{
    backend::Cuda,
    module::Module,
    nn::interpolate::{Interpolate2dConfig, InterpolateMode},
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use burn_depth_pro::{
    inference::rgb_to_input_tensor,
    model::depth_pro::{DepthPro, DepthProConfig, HeadDebug, layers::encoder::EncoderDebug},
};
use image::GenericImageView;
use safetensors::tensor::{SafeTensors, TensorView};

type CorrectnessBackend = Cuda<f32>;

#[derive(Clone)]
struct FeatureTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

fn flat_index_to_coords(index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    let mut remainder = index;
    for dim in 0..shape.len() {
        let stride = shape[dim + 1..].iter().product::<usize>().max(1);
        let coord = remainder / stride;
        coords.push(coord);
        remainder %= stride;
    }
    coords
}

struct TorchReference {
    depth: Vec<f32>,
    depth_shape: Vec<usize>,
    fovx: f32,
    fovy: f32,
    encoder_features: Vec<FeatureTensor>,
    decoder_fusions: Vec<FeatureTensor>,
    encoder_merge_latent0: Option<FeatureTensor>,
    encoder_merge_latent1: Option<FeatureTensor>,
    encoder_latent0_tokens: Option<FeatureTensor>,
    encoder_latent1_tokens: Option<FeatureTensor>,
    encoder_latent0_merge_input: Option<FeatureTensor>,
    encoder_latent1_merge_input: Option<FeatureTensor>,
    encoder_merge_x0: Option<FeatureTensor>,
    encoder_merge_x1: Option<FeatureTensor>,
    encoder_merge_x2: Option<FeatureTensor>,
    canonical_inverse_depth: Option<FeatureTensor>,
    decoder_feature: Option<FeatureTensor>,
    decoder_lowres_feature: Option<FeatureTensor>,
    head_conv0: Option<FeatureTensor>,
    head_deconv: Option<FeatureTensor>,
    head_conv1: Option<FeatureTensor>,
    head_relu: Option<FeatureTensor>,
    head_pre_out: Option<FeatureTensor>,
}

struct BurnOutputs {
    depth: Vec<f32>,
    height: usize,
    width: usize,
    fovx: f32,
    fovy: f32,
    encoder_features: Vec<FeatureTensor>,
    decoder_fusions: Vec<FeatureTensor>,
    encoder_merge_latent0: FeatureTensor,
    encoder_merge_latent1: FeatureTensor,
    encoder_latent0_tokens: FeatureTensor,
    encoder_latent1_tokens: FeatureTensor,
    encoder_latent0_merge_input: FeatureTensor,
    encoder_latent1_merge_input: FeatureTensor,
    encoder_merge_x0: FeatureTensor,
    encoder_merge_x1: FeatureTensor,
    encoder_merge_x2: FeatureTensor,
    canonical_inverse_depth: FeatureTensor,
    decoder_feature: FeatureTensor,
    decoder_lowres_feature: FeatureTensor,
    head_conv0: FeatureTensor,
    head_deconv: FeatureTensor,
    head_conv1: FeatureTensor,
    head_relu: FeatureTensor,
    head_pre_out: FeatureTensor,
}

fn tensor_view_to_vec(view: &TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(bytes)
        })
        .collect()
}

fn load_torch_reference(path: &Path) -> Result<TorchReference, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let depth_view = tensors
        .tensor("metric_depth")
        .map_err(|_| "missing `metric_depth` tensor in reference file")?;
    let depth_shape = depth_view.shape().to_vec();
    if depth_shape.len() != 3 || depth_shape[2] != 1 {
        return Err(format!(
            "expected torch depth shape [H, W, 1], got {:?}",
            depth_shape
        )
        .into());
    }
    let depth = tensor_view_to_vec(&depth_view);

    let fovy_view = tensors
        .tensor("fovy")
        .map_err(|_| "missing `fovy` tensor in reference file")?;
    if fovy_view.shape() != [1] {
        return Err(format!("expected fovy shape [1], got {:?}", fovy_view.shape()).into());
    }
    let fovy = tensor_view_to_vec(&fovy_view)[0];

    let fovx_view = tensors
        .tensor("fovx")
        .map_err(|_| "missing `fovx` tensor in reference file")?;
    if fovx_view.shape() != [1] {
        return Err(format!("expected fovx shape [1], got {:?}", fovx_view.shape()).into());
    }
    let fovx = tensor_view_to_vec(&fovx_view)[0];

    let mut encoder_features = Vec::new();
    for idx in 0.. {
        let key = format!("encoder_feature_{}", idx);
        match tensors.tensor(&key) {
            Ok(view) => {
                let shape = view.shape().to_vec();
                let data = tensor_view_to_vec(&view);
                encoder_features.push(FeatureTensor { data, shape });
            }
            Err(_) => break,
        }
    }

    let mut decoder_fusions = Vec::new();
    for idx in 0.. {
        let key = format!("decoder_fusion_{}", idx);
        match tensors.tensor(&key) {
            Ok(view) => {
                let shape = view.shape().to_vec();
                let data = tensor_view_to_vec(&view);
                decoder_fusions.push(FeatureTensor { data, shape });
            }
            Err(_) => break,
        }
    }

    let load_optional_feature = |name: &str| -> Option<FeatureTensor> {
        tensors.tensor(name).ok().map(|view| FeatureTensor {
            data: tensor_view_to_vec(&view),
            shape: view.shape().to_vec(),
        })
    };

    Ok(TorchReference {
        depth,
        depth_shape,
        fovx,
        fovy,
        encoder_features,
        decoder_fusions,
        encoder_merge_latent0: load_optional_feature("encoder_merge_latent0"),
        encoder_merge_latent1: load_optional_feature("encoder_merge_latent1"),
        encoder_latent0_tokens: load_optional_feature("encoder_latent0_tokens"),
        encoder_latent1_tokens: load_optional_feature("encoder_latent1_tokens"),
        encoder_latent0_merge_input: load_optional_feature("encoder_latent0_merge_input"),
        encoder_latent1_merge_input: load_optional_feature("encoder_latent1_merge_input"),
        encoder_merge_x0: load_optional_feature("encoder_merge_x0"),
        encoder_merge_x1: load_optional_feature("encoder_merge_x1"),
        encoder_merge_x2: load_optional_feature("encoder_merge_x2"),
        canonical_inverse_depth: load_optional_feature("canonical_inverse_depth"),
        decoder_feature: load_optional_feature("decoder_feature"),
        decoder_lowres_feature: load_optional_feature("decoder_lowres_feature"),
        head_conv0: load_optional_feature("head_conv0"),
        head_deconv: load_optional_feature("head_deconv"),
        head_conv1: load_optional_feature("head_conv1"),
        head_relu: load_optional_feature("head_relu"),
        head_pre_out: load_optional_feature("head_pre_out"),
    })
}

fn tensor_to_feature(
    tensor: Tensor<CorrectnessBackend, 4>,
) -> Result<FeatureTensor, Box<dyn std::error::Error>> {
    let shape = tensor.shape().dims::<4>().to_vec();
    let data = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to extract encoder feature: {err:?}"))?;
    Ok(FeatureTensor { data, shape })
}

fn feature_tensor_to_tensor(
    feature: &FeatureTensor,
    device: &<CorrectnessBackend as Backend>::Device,
) -> Result<Tensor<CorrectnessBackend, 4>, Box<dyn std::error::Error>> {
    let dims: [usize; 4] = feature
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| format!("expected 4d feature shape, got {:?}", feature.shape))?;
    let dims_i32 = [
        dims[0] as i32,
        dims[1] as i32,
        dims[2] as i32,
        dims[3] as i32,
    ];
    Ok(
        Tensor::<CorrectnessBackend, 1>::from_floats(feature.data.as_slice(), device)
            .reshape(dims_i32),
    )
}

fn compute_burn_outputs(image_path: &Path) -> Result<BurnOutputs, Box<dyn std::error::Error>> {
    let device = <CorrectnessBackend as Backend>::Device::default();
    let checkpoint = Path::new("assets/model/depth_pro.mpk");

    let model = DepthPro::<CorrectnessBackend>::new(&device, DepthProConfig::default());
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(checkpoint, &recorder, &device)
        .map_err(|err| format!("failed to load checkpoint: {err}"))?;

    let image = image::open(image_path)?;
    let (orig_width, orig_height) = image.dimensions();
    let image = image.to_rgb8();

    let width = orig_width as usize;
    let height = orig_height as usize;
    let input = rgb_to_input_tensor::<CorrectnessBackend>(image.as_raw(), width, height, &device)
        .map_err(|err| format!("failed to prepare input tensor: {err}"))?;

    let mut feature_input = input.clone();
    let img_size = model.img_size();
    if height != img_size || width != img_size {
        let resize = Interpolate2dConfig::new()
            .with_output_size(Some([img_size, img_size]))
            .with_mode(InterpolateMode::Linear)
            .init();
        feature_input = resize.forward(feature_input);
    }

    let EncoderDebug {
        features: encoder_feature_tensors,
        latent0: encoder_merge_latent0_tensor,
        latent1: encoder_merge_latent1_tensor,
        latent0_tokens,
        latent1_tokens,
        latent0_merge_input,
        latent1_merge_input,
        x0_tokens,
        x1_tokens: _,
        x2_tokens: _,
        split_x0: _,
        split_x1: _,
        split_x2: _,
        merged_x0: encoder_merge_x0_tensor,
        merged_x1: encoder_merge_x1_tensor,
        merged_x2: encoder_merge_x2_tensor,
    }: EncoderDebug<CorrectnessBackend> = model.encoder_forward_debug(feature_input.clone());
    println!("Burn encoder feature shapes:");
    let mut burn_features = Vec::with_capacity(encoder_feature_tensors.len());
    for (idx, feature) in encoder_feature_tensors.into_iter().enumerate() {
        let feature: Tensor<CorrectnessBackend, 4> = feature;
        let dims = feature.shape().dims::<4>();
        println!("  {idx}: {:?}", dims);
        burn_features.push(tensor_to_feature(feature)?);
    }
    let encoder_merge_latent0 = tensor_to_feature(encoder_merge_latent0_tensor)?;
    let encoder_merge_latent1 = tensor_to_feature(encoder_merge_latent1_tensor)?;
    let encoder_latent0_merge_input = tensor_to_feature(latent0_merge_input.clone())
        .map_err(|err| format!("failed to fetch encoder latent0 merge input: {err}"))?;
    let encoder_latent1_merge_input = tensor_to_feature(latent1_merge_input.clone())
        .map_err(|err| format!("failed to fetch encoder latent1 merge input: {err}"))?;
    let high_count = x0_tokens.shape().dims::<4>()[0];
    let latent0_subset = latent0_tokens.clone().slice([
        0..high_count,
        0..latent0_tokens.shape().dims::<4>()[1],
        0..latent0_tokens.shape().dims::<4>()[2],
        0..latent0_tokens.shape().dims::<4>()[3],
    ]);
    let latent1_subset = latent1_tokens.clone().slice([
        0..high_count,
        0..latent1_tokens.shape().dims::<4>()[1],
        0..latent1_tokens.shape().dims::<4>()[2],
        0..latent1_tokens.shape().dims::<4>()[3],
    ]);
    let encoder_latent0_tokens = tensor_to_feature(latent0_subset)
        .map_err(|err| format!("failed to fetch encoder latent0 tokens: {err}"))?;
    let encoder_latent1_tokens = tensor_to_feature(latent1_subset)
        .map_err(|err| format!("failed to fetch encoder latent1 tokens: {err}"))?;
    let encoder_merge_x0 = tensor_to_feature(encoder_merge_x0_tensor)?;
    let encoder_merge_x1 = tensor_to_feature(encoder_merge_x1_tensor)?;
    let encoder_merge_x2 = tensor_to_feature(encoder_merge_x2_tensor)?;
    let (
        _canonical_decoder,
        decoder_feature_tensor,
        decoder_lowres_tensor,
        decoder_fusion_tensors,
        _,
    ) = model.forward_with_decoder(feature_input.clone());
    let head_debug = model.head_debug(decoder_feature_tensor.clone());
    let HeadDebug {
        conv0,
        deconv,
        conv1,
        relu,
        pre_out,
        canonical,
    } = head_debug;
    let canonical_tensor = canonical.clone();
    let canonical_shape = canonical_tensor.shape().dims::<4>().to_vec();
    let canonical_data = canonical_tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to fetch canonical inverse depth: {err:?}"))?;
    let (mut canon_min, mut canon_max, mut canon_sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32);
    for value in canonical_data.iter() {
        canon_min = canon_min.min(*value);
        canon_max = canon_max.max(*value);
        canon_sum += *value;
    }
    let canon_mean = canon_sum / canonical_data.len() as f32;
    println!(
        "Burn canonical inverse depth stats: min={canon_min:.6}, max={canon_max:.6}, mean={canon_mean:.6}"
    );
    let canonical_feature = FeatureTensor {
        data: canonical_data,
        shape: canonical_shape,
    };

    let decoder_feature = tensor_to_feature(decoder_feature_tensor)
        .map_err(|err| format!("failed to fetch decoder feature: {err}"))?;
    let decoder_lowres_feature = tensor_to_feature(decoder_lowres_tensor)
        .map_err(|err| format!("failed to fetch decoder lowres feature: {err}"))?;
    let mut decoder_fusions = Vec::with_capacity(decoder_fusion_tensors.len());
    for (idx, fusion) in decoder_fusion_tensors.into_iter().enumerate() {
        decoder_fusions.push(
            tensor_to_feature(fusion)
                .map_err(|err| format!("failed to fetch decoder fusion {idx}: {err}"))?,
        );
    }
    let head_conv0 =
        tensor_to_feature(conv0).map_err(|err| format!("failed to fetch head conv0: {err}"))?;
    let head_deconv =
        tensor_to_feature(deconv).map_err(|err| format!("failed to fetch head deconv: {err}"))?;
    let head_conv1 =
        tensor_to_feature(conv1).map_err(|err| format!("failed to fetch head conv1: {err}"))?;
    let head_relu =
        tensor_to_feature(relu).map_err(|err| format!("failed to fetch head relu: {err}"))?;
    let head_pre_out =
        tensor_to_feature(pre_out).map_err(|err| format!("failed to fetch head pre_out: {err}"))?;

    let output = model.infer(input, None, InterpolateMode::Linear);
    let depth = output
        .depth
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to fetch burn depth: {err:?}"))?;

    let f_px = output
        .focallength_px
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to fetch burn focal length: {err:?}"))?[0];
    println!("Burn focal length (px): {f_px:.6}");
    let width_f = orig_width as f32;
    let height_f = orig_height as f32;
    let fovx_rad = 2.0 * (0.5 * width_f / f_px).atan();
    let fovx = fovx_rad * 180.0 / PI;
    let half_fovx = 0.5 * fovx_rad;
    let fovy_rad = 2.0 * ((height_f / width_f) * half_fovx.tan()).atan();
    let fovy = fovy_rad * 180.0 / PI;
    println!("Burn FOVx: {fovx:.6} deg");
    println!("Burn FOVy: {fovy:.6} deg");

    Ok(BurnOutputs {
        depth,
        height,
        width,
        fovx,
        fovy,
        encoder_features: burn_features,
        decoder_fusions,
        encoder_merge_latent0,
        encoder_merge_latent1,
        encoder_latent0_tokens,
        encoder_latent1_tokens,
        encoder_latent0_merge_input,
        encoder_latent1_merge_input,
        encoder_merge_x0,
        encoder_merge_x1,
        encoder_merge_x2,
        canonical_inverse_depth: canonical_feature,
        decoder_feature,
        decoder_lowres_feature,
        head_conv0,
        head_deconv,
        head_conv1,
        head_relu,
        head_pre_out,
    })
}

fn compute_stats(burn: &[f32], reference: &[f32]) -> (f32, f32, f32) {
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;

    for (&burn_value, &reference_value) in burn.iter().zip(reference.iter()) {
        let diff = burn_value - reference_value;
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);

        let rel = if reference_value.abs() > 1e-6 {
            abs / reference_value.abs()
        } else {
            0.0
        };
        max_rel = max_rel.max(rel);
    }

    let mean_abs = sum_abs / burn.len() as f32;
    (mean_abs, max_abs, max_rel)
}

fn compare_decoder_with_reference(
    torch_reference: &TorchReference,
) -> Result<(), Box<dyn std::error::Error>> {
    if torch_reference.encoder_features.is_empty() {
        println!("Torch reference missing encoder features; skipping decoder replay.");
        return Ok(());
    }

    if std::env::var("BURN_SKIP_DECODER_REPLAY").is_ok() {
        println!("Skipping decoder replay (BURN_SKIP_DECODER_REPLAY set).");
        return Ok(());
    }

    let device = <CorrectnessBackend as Backend>::Device::default();
    let checkpoint = Path::new("assets/model/depth_pro.mpk");

    let base_model = DepthPro::<CorrectnessBackend>::new(&device, DepthProConfig::default());
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = base_model
        .load_file(checkpoint, &recorder, &device)
        .map_err(|err| format!("failed to load checkpoint for decoder replay: {err}"))?;

    let mut encoder_inputs = Vec::with_capacity(torch_reference.encoder_features.len());
    for feature in torch_reference.encoder_features.iter() {
        encoder_inputs.push(feature_tensor_to_tensor(feature, &device)?);
    }

    let (replay_feature, replay_lowres, replay_fusions) =
        model.decoder_from_features(encoder_inputs.as_slice());

    let burn_replay_feature = tensor_to_feature(replay_feature)?;
    let burn_replay_lowres = tensor_to_feature(replay_lowres)?;
    let mut burn_replay_fusions = Vec::with_capacity(replay_fusions.len());
    for fusion in replay_fusions {
        burn_replay_fusions.push(tensor_to_feature(fusion)?);
    }

    let report_feature = |label: &str, burn: &FeatureTensor, torch_opt: &Option<FeatureTensor>| {
        if let Some(torch_feat) = torch_opt.as_ref() {
            if burn.shape != torch_feat.shape {
                println!(
                    "[Replay] {label} shape mismatch: torch {:?}, burn {:?}",
                    torch_feat.shape, burn.shape
                );
            } else {
                let (mean_abs, max_abs, max_rel) =
                    compute_stats(burn.data.as_slice(), torch_feat.data.as_slice());
                println!(
                    "[Replay] {label}: mean abs={mean_abs:.6}, max abs={max_abs:.6}, max rel={max_rel:.6}"
                );
                if max_abs > 1e-3 {
                    let mut max_diff = 0.0f32;
                    let mut max_index = 0usize;
                    let mut max_pair = (0.0f32, 0.0f32);
                    for (idx, (&burn_value, &torch_value)) in
                        burn.data.iter().zip(torch_feat.data.iter()).enumerate()
                    {
                        let diff = (burn_value - torch_value).abs();
                        if diff > max_diff {
                            max_diff = diff;
                            max_index = idx;
                            max_pair = (burn_value, torch_value);
                        }
                    }
                    let coords = flat_index_to_coords(max_index, burn.shape.as_slice());
                    println!(
                        "[Replay] {label} max diff at {:?}: burn={:.6}, torch={:.6}, diff={:.6}",
                        coords, max_pair.0, max_pair.1, max_diff
                    );
                }
            }
        } else {
            println!("[Replay] Torch reference missing {label}; skipping.");
        }
    };

    report_feature(
        "Decoder feature",
        &burn_replay_feature,
        &torch_reference.decoder_feature,
    );
    report_feature(
        "Decoder lowres feature",
        &burn_replay_lowres,
        &torch_reference.decoder_lowres_feature,
    );

    if burn_replay_fusions.len() == torch_reference.decoder_fusions.len() {
        for (idx, (burn_fusion, torch_fusion)) in burn_replay_fusions
            .iter()
            .zip(torch_reference.decoder_fusions.iter())
            .enumerate()
        {
            if burn_fusion.shape != torch_fusion.shape {
                println!(
                    "[Replay] Decoder fusion {idx} shape mismatch: torch {:?}, burn {:?}",
                    torch_fusion.shape, burn_fusion.shape
                );
                continue;
            }
            let (mean_abs, max_abs, max_rel) =
                compute_stats(burn_fusion.data.as_slice(), torch_fusion.data.as_slice());
            println!(
                "[Replay] Decoder fusion {idx}: mean abs={mean_abs:.6}, max abs={max_abs:.6}, max rel={max_rel:.6}"
            );
            if max_abs > 1e-3 {
                let mut max_diff = 0.0f32;
                let mut max_index = 0usize;
                let mut max_pair = (0.0f32, 0.0f32);
                for (flat_index, (&burn_value, &torch_value)) in burn_fusion
                    .data
                    .iter()
                    .zip(torch_fusion.data.iter())
                    .enumerate()
                {
                    let diff = (burn_value - torch_value).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_index = flat_index;
                        max_pair = (burn_value, torch_value);
                    }
                }
                let coords = flat_index_to_coords(max_index, burn_fusion.shape.as_slice());
                println!(
                    "[Replay] Decoder fusion {idx} max diff at {:?}: burn={:.6}, torch={:.6}, diff={:.6}",
                    coords, max_pair.0, max_pair.1, max_diff
                );
            }
        }
    } else {
        println!(
            "[Replay] Decoder fusion count mismatch: torch={}, burn={}",
            torch_reference.decoder_fusions.len(),
            burn_replay_fusions.len()
        );
    }

    Ok(())
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

    let torch_reference = load_torch_reference(reference_path)?;
    compare_decoder_with_reference(&torch_reference)?;
    let burn_outputs = compute_burn_outputs(image_path)?;

    if burn_outputs.height != torch_reference.depth_shape[0]
        || burn_outputs.width != torch_reference.depth_shape[1]
    {
        return Err(format!(
            "shape mismatch: torch {}x{}, burn {}x{}",
            torch_reference.depth_shape[0],
            torch_reference.depth_shape[1],
            burn_outputs.height,
            burn_outputs.width
        )
        .into());
    }

    let (depth_mean_abs, depth_max_abs, depth_max_rel) =
        compute_stats(&burn_outputs.depth, &torch_reference.depth);
    let fovx_diff = (burn_outputs.fovx - torch_reference.fovx).abs();
    let fovy_diff = (burn_outputs.fovy - torch_reference.fovy).abs();

    let (burn_min, burn_max) = burn_outputs.depth.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(min_v, max_v), value| (min_v.min(*value), max_v.max(*value)),
    );
    let (torch_min, torch_max) = torch_reference.depth.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(min_v, max_v), value| (min_v.min(*value), max_v.max(*value)),
    );

    println!("Pixel mean abs diff: {depth_mean_abs:.6}");
    println!("Pixel max abs diff:  {depth_max_abs:.6}");
    println!("Pixel max rel diff:  {depth_max_rel:.6}");
    println!("FOVx difference:     {fovx_diff:.6} deg");
    println!("FOVy difference:     {fovy_diff:.6} deg");
    println!(
        "Depth ranges: burn=({burn_min:.3}, {burn_max:.3}) ft, torch=({torch_min:.3}, {torch_max:.3}) ft"
    );

    if let Some(canonical_ref) = torch_reference.canonical_inverse_depth.as_ref() {
        if burn_outputs.canonical_inverse_depth.shape != canonical_ref.shape {
            println!(
                "Canonical inverse depth shape mismatch: torch {:?}, burn {:?}",
                canonical_ref.shape, burn_outputs.canonical_inverse_depth.shape
            );
        } else {
            let mut diff_sum = 0.0f32;
            let mut min_diff = f32::INFINITY;
            let mut max_diff = f32::NEG_INFINITY;
            for (&burn_value, &torch_value) in burn_outputs
                .canonical_inverse_depth
                .data
                .iter()
                .zip(canonical_ref.data.iter())
            {
                let diff = burn_value - torch_value;
                diff_sum += diff;
                min_diff = min_diff.min(diff);
                max_diff = max_diff.max(diff);
            }
            let mean_diff = diff_sum / burn_outputs.canonical_inverse_depth.data.len() as f32;
            let (canon_mean_abs, canon_max_abs, canon_max_rel) = compute_stats(
                &burn_outputs.canonical_inverse_depth.data,
                &canonical_ref.data,
            );
            println!(
                "Canonical inverse depth diff: mean abs={canon_mean_abs:.6}, max abs={canon_max_abs:.6}, max rel={canon_max_rel:.6}"
            );
            println!("Canonical inverse depth mean diff: {mean_diff:.6}");
            println!("Canonical inverse depth diff range: min={min_diff:.6}, max={max_diff:.6}");
        }
    } else {
        println!("Torch reference missing canonical inverse depth; skipping comparison.");
    }

    if let Some(decoder_ref) = torch_reference.decoder_feature.as_ref() {
        if burn_outputs.decoder_feature.shape != decoder_ref.shape {
            println!(
                "Decoder feature shape mismatch: torch {:?}, burn {:?}",
                decoder_ref.shape, burn_outputs.decoder_feature.shape
            );
        } else {
            let mut diff_sum = 0.0f32;
            let mut min_diff = f32::INFINITY;
            let mut max_diff = f32::NEG_INFINITY;
            let mut max_abs_seen = 0.0f32;
            let mut max_index = 0usize;
            let mut max_pair = (0.0f32, 0.0f32);
            for (idx, (&burn_value, &torch_value)) in burn_outputs
                .decoder_feature
                .data
                .iter()
                .zip(decoder_ref.data.iter())
                .enumerate()
            {
                let diff = burn_value - torch_value;
                diff_sum += diff;
                min_diff = min_diff.min(diff);
                max_diff = max_diff.max(diff);
                let abs = diff.abs();
                if abs > max_abs_seen {
                    max_abs_seen = abs;
                    max_index = idx;
                    max_pair = (burn_value, torch_value);
                }
            }
            let mean_diff = diff_sum / burn_outputs.decoder_feature.data.len() as f32;
            let (mean_abs, max_abs, max_rel) =
                compute_stats(&burn_outputs.decoder_feature.data, &decoder_ref.data);
            println!(
                "Decoder feature diff: mean abs={mean_abs:.6}, max abs={max_abs:.6}, max rel={max_rel:.6}"
            );
            println!(
                "Decoder feature mean diff={mean_diff:.6}, range=({min_diff:.6}, {max_diff:.6})"
            );
            if max_abs > 1e-3 {
                let coords =
                    flat_index_to_coords(max_index, burn_outputs.decoder_feature.shape.as_slice());
                println!(
                    "Decoder feature max diff at {:?}: burn={:.6}, torch={:.6}, diff={:.6}",
                    coords, max_pair.0, max_pair.1, max_abs
                );
            }
        }
    } else {
        println!("Torch reference missing decoder feature; skipping comparison.");
    }

    if let Some(decoder_lowres_ref) = torch_reference.decoder_lowres_feature.as_ref() {
        if burn_outputs.decoder_lowres_feature.shape != decoder_lowres_ref.shape {
            println!(
                "Decoder lowres feature shape mismatch: torch {:?}, burn {:?}",
                decoder_lowres_ref.shape, burn_outputs.decoder_lowres_feature.shape
            );
        } else {
            let mut diff_sum = 0.0f32;
            let mut min_diff = f32::INFINITY;
            let mut max_diff = f32::NEG_INFINITY;
            let mut max_abs_seen = 0.0f32;
            let mut max_index = 0usize;
            let mut max_pair = (0.0f32, 0.0f32);
            for (idx, (&burn_value, &torch_value)) in burn_outputs
                .decoder_lowres_feature
                .data
                .iter()
                .zip(decoder_lowres_ref.data.iter())
                .enumerate()
            {
                let diff = burn_value - torch_value;
                diff_sum += diff;
                min_diff = min_diff.min(diff);
                max_diff = max_diff.max(diff);
                let abs = diff.abs();
                if abs > max_abs_seen {
                    max_abs_seen = abs;
                    max_index = idx;
                    max_pair = (burn_value, torch_value);
                }
            }
            let mean_diff = diff_sum / burn_outputs.decoder_lowres_feature.data.len() as f32;
            let (mean_abs, max_abs, max_rel) = compute_stats(
                &burn_outputs.decoder_lowres_feature.data,
                &decoder_lowres_ref.data,
            );
            println!(
                "Decoder lowres feature diff: mean abs={mean_abs:.6}, max abs={max_abs:.6}, max rel={max_rel:.6}"
            );
            println!(
                "Decoder lowres feature mean diff={mean_diff:.6}, range=({min_diff:.6}, {max_diff:.6})"
            );
            if max_abs > 1e-3 {
                let coords = flat_index_to_coords(
                    max_index,
                    burn_outputs.decoder_lowres_feature.shape.as_slice(),
                );
                println!(
                    "Decoder lowres max diff at {:?}: burn={:.6}, torch={:.6}, diff={:.6}",
                    coords, max_pair.0, max_pair.1, max_abs
                );
            }
        }
    } else {
        println!("Torch reference missing decoder lowres feature; skipping comparison.");
    }

    if burn_outputs.decoder_fusions.len() == torch_reference.decoder_fusions.len() {
        for (idx, (burn_fusion, torch_fusion)) in burn_outputs
            .decoder_fusions
            .iter()
            .zip(torch_reference.decoder_fusions.iter())
            .enumerate()
        {
            if burn_fusion.shape != torch_fusion.shape {
                println!(
                    "Decoder fusion {idx} shape mismatch: torch {:?}, burn {:?}",
                    torch_fusion.shape, burn_fusion.shape
                );
                continue;
            }

            let mut diff_sum = 0.0f32;
            let mut min_diff = f32::INFINITY;
            let mut max_diff = f32::NEG_INFINITY;
            let mut max_abs_seen = 0.0f32;
            let mut max_index = 0usize;
            let mut max_pair = (0.0f32, 0.0f32);
            for (flat_index, (&burn_value, &torch_value)) in burn_fusion
                .data
                .iter()
                .zip(torch_fusion.data.iter())
                .enumerate()
            {
                let diff = burn_value - torch_value;
                diff_sum += diff;
                min_diff = min_diff.min(diff);
                max_diff = max_diff.max(diff);
                let abs = diff.abs();
                if abs > max_abs_seen {
                    max_abs_seen = abs;
                    max_index = flat_index;
                    max_pair = (burn_value, torch_value);
                }
            }
            let mean_diff = diff_sum / burn_fusion.data.len() as f32;
            let (mean_abs, max_abs, max_rel) =
                compute_stats(burn_fusion.data.as_slice(), torch_fusion.data.as_slice());
            println!(
                "Decoder fusion {idx}: mean abs={mean_abs:.6}, max abs={max_abs:.6}, max rel={max_rel:.6}"
            );
            println!(
                "Decoder fusion {idx}: mean diff={mean_diff:.6}, range=({min_diff:.6}, {max_diff:.6})"
            );
            if max_abs > 1e-3 {
                let coords = flat_index_to_coords(max_index, burn_fusion.shape.as_slice());
                println!(
                    "Decoder fusion {idx} max diff at {:?}: burn={:.6}, torch={:.6}, diff={:.6}",
                    coords, max_pair.0, max_pair.1, max_abs
                );
            }
        }
    } else {
        println!(
            "Skipping decoder fusion comparison (torch={}, burn={}).",
            torch_reference.decoder_fusions.len(),
            burn_outputs.decoder_fusions.len()
        );
    }

    let compare_feature = |name: &str, burn: &FeatureTensor, torch_opt: &Option<FeatureTensor>| {
        if let Some(torch_feat) = torch_opt.as_ref() {
            if burn.shape != torch_feat.shape {
                println!(
                    "{name} shape mismatch: torch {:?}, burn {:?}",
                    torch_feat.shape, burn.shape
                );
            } else {
                let mut diff_sum = 0.0f32;
                let mut min_diff = f32::INFINITY;
                let mut max_diff = f32::NEG_INFINITY;
                let mut max_abs_seen = 0.0f32;
                let mut max_index = 0usize;
                let mut max_pair = (0.0f32, 0.0f32);
                for (idx, (&burn_value, &torch_value)) in
                    burn.data.iter().zip(torch_feat.data.iter()).enumerate()
                {
                    let diff = burn_value - torch_value;
                    diff_sum += diff;
                    min_diff = min_diff.min(diff);
                    max_diff = max_diff.max(diff);
                    let abs = diff.abs();
                    if abs > max_abs_seen {
                        max_abs_seen = abs;
                        max_index = idx;
                        max_pair = (burn_value, torch_value);
                    }
                }
                let mean_diff = diff_sum / burn.data.len() as f32;
                let (mean_abs, max_abs, max_rel) =
                    compute_stats(burn.data.as_slice(), torch_feat.data.as_slice());
                println!(
                    "{name}: mean abs={mean_abs:.6}, max abs={max_abs:.6}, max rel={max_rel:.6}"
                );
                println!("{name}: mean diff={mean_diff:.6}, range=({min_diff:.6}, {max_diff:.6})");
                if max_abs > 1e-3 {
                    let coords = flat_index_to_coords(max_index, burn.shape.as_slice());
                    println!(
                        "{name}: max abs diff at {:?}: burn={:.6}, torch={:.6}, diff={:.6}",
                        coords, max_pair.0, max_pair.1, max_abs
                    );
                }
            }
        } else {
            println!("Torch reference missing {name}; skipping comparison.");
        }
    };

    compare_feature(
        "Encoder latent0 tokens",
        &burn_outputs.encoder_latent0_tokens,
        &torch_reference.encoder_latent0_tokens,
    );
    compare_feature(
        "Encoder latent1 tokens",
        &burn_outputs.encoder_latent1_tokens,
        &torch_reference.encoder_latent1_tokens,
    );
    compare_feature(
        "Encoder latent0 merge input",
        &burn_outputs.encoder_latent0_merge_input,
        &torch_reference.encoder_latent0_merge_input,
    );
    compare_feature(
        "Encoder latent1 merge input",
        &burn_outputs.encoder_latent1_merge_input,
        &torch_reference.encoder_latent1_merge_input,
    );
    compare_feature(
        "Encoder merge latent0",
        &burn_outputs.encoder_merge_latent0,
        &torch_reference.encoder_merge_latent0,
    );
    compare_feature(
        "Encoder merge latent1",
        &burn_outputs.encoder_merge_latent1,
        &torch_reference.encoder_merge_latent1,
    );
    compare_feature(
        "Encoder merge x0",
        &burn_outputs.encoder_merge_x0,
        &torch_reference.encoder_merge_x0,
    );
    compare_feature(
        "Encoder merge x1",
        &burn_outputs.encoder_merge_x1,
        &torch_reference.encoder_merge_x1,
    );
    compare_feature(
        "Encoder merge x2",
        &burn_outputs.encoder_merge_x2,
        &torch_reference.encoder_merge_x2,
    );
    compare_feature(
        "Head conv0 feature",
        &burn_outputs.head_conv0,
        &torch_reference.head_conv0,
    );
    compare_feature(
        "Head deconv feature",
        &burn_outputs.head_deconv,
        &torch_reference.head_deconv,
    );
    compare_feature(
        "Head conv1 feature",
        &burn_outputs.head_conv1,
        &torch_reference.head_conv1,
    );
    compare_feature(
        "Head relu feature",
        &burn_outputs.head_relu,
        &torch_reference.head_relu,
    );
    compare_feature(
        "Head pre_out feature",
        &burn_outputs.head_pre_out,
        &torch_reference.head_pre_out,
    );

    let mut feature_within_tolerance = true;
    if burn_outputs.encoder_features.len() == torch_reference.encoder_features.len() {
        for (idx, (burn_feature, torch_feature)) in burn_outputs
            .encoder_features
            .iter()
            .zip(torch_reference.encoder_features.iter())
            .enumerate()
        {
            if burn_feature.shape != torch_feature.shape {
                return Err(format!(
                    "encoder feature {idx} shape mismatch: torch {:?}, burn {:?}",
                    torch_feature.shape, burn_feature.shape
                )
                .into());
            }

            let (mean_abs, max_abs, max_rel) =
                compute_stats(&burn_feature.data, &torch_feature.data);
            println!(
                "Feature {idx}: mean abs={mean_abs:.6}, max abs={max_abs:.6}, max rel={max_rel:.6}"
            );

            if max_abs > 5e-3 || mean_abs > 1e-3 || max_rel > 5e-3 {
                feature_within_tolerance = false;
            }
        }
    } else {
        println!(
            "Skipping encoder feature comparison (torch={}, burn={}).",
            torch_reference.encoder_features.len(),
            burn_outputs.encoder_features.len()
        );
    }

    const DEPTH_MAX_ABS_THRESHOLD: f32 = 5e-3;
    const DEPTH_MEAN_ABS_THRESHOLD: f32 = 1e-3;
    const DEPTH_MAX_REL_THRESHOLD: f32 = 5e-3;
    const FOVX_THRESHOLD: f32 = 1e-3;
    const FOVY_THRESHOLD: f32 = 1e-3;

    let depth_ok = depth_max_abs <= DEPTH_MAX_ABS_THRESHOLD
        && depth_mean_abs <= DEPTH_MEAN_ABS_THRESHOLD
        && depth_max_rel <= DEPTH_MAX_REL_THRESHOLD;
    let fovx_ok = fovx_diff <= FOVX_THRESHOLD;
    let fovy_ok = fovy_diff <= FOVY_THRESHOLD;

    if depth_ok && fovx_ok && fovy_ok && feature_within_tolerance {
        println!("Burn output matches Torch reference within tolerance.");
        Ok(())
    } else {
        Err("Burn output deviates from Torch reference beyond tolerance.".into())
    }
}
