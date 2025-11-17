use std::{
    env,
    path::{Path, PathBuf},
};

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn_depth::{
    inference::rgb_to_input_tensor,
    model::{
        depth_anything3::{DepthAnything3, DepthAnything3Config, DepthTrace},
        prepare_depth_anything3_image,
    },
};
use image::RgbImage;
use safetensors::tensor::{SafeTensors, TensorView};

type NdBackend = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = std::thread::Builder::new()
        .name("da3_small_correctness".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| run().map_err(|err| err.to_string()))
        .map_err(|err| format!("Failed to spawn correctness thread: {err}"))?;
    handle
        .join()
        .map_err(|_| -> Box<dyn std::error::Error> { "Correctness thread panicked".into() })?
        .map_err(|err: String| err.into())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args();
    let _exe = args.next();
    let reference_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/image/test_da3_small_reference.safetensors"));
    let image_path = args
        .next()
        .unwrap_or_else(|| "assets/image/test.jpg".to_string());

    let device = NdArrayDevice::default();
    println!("Loading Depth Anything 3 small checkpoint...");
    let config = DepthAnything3Config::metric_small();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = DepthAnything3::<NdBackend>::new(&device, config)
        .load_file("assets/model/da3_small.mpk", &recorder, &device)
        .map_err(|err| format!("Failed to load DA3 small checkpoint: {err}"))?;

    println!(
        "Checkpoint ready; loading reference tensors from {}...",
        reference_path.display()
    );
    let reference = load_reference(&reference_path)?;
    println!(
        "Reference tensors loaded (depth {:?}); preparing input tensor...",
        reference.depth.shape
    );
    let input_tensor = if let Some(metric_input) = &reference.metric_input {
        println!(
            "Using embedded metric_input tensor from {}",
            reference_path.display()
        );
        tensor_from_reference::<4>(metric_input, &device)
            .map_err(|err| format!("Failed to build reference input tensor: {err}"))?
    } else {
        println!(
            "Reference `{}` lacks metric_input; falling back to prepared RGB `{image_path}`.",
            reference_path.display()
        );
        let prepared = load_prepared_image(&image_path, model.img_size())?;
        rgb_to_input_tensor::<NdBackend>(
            prepared.rgb.as_raw(),
            prepared.width,
            prepared.height,
            &device,
        )?
    };

    println!("Running DA3 small inference...");
    let (refined, trace) = model.infer_with_trace(input_tensor);

    println!("Inference finished; analyzing outputs...");
    analyze_tensor3("final depth", Some(&refined.depth), Some(&reference.depth));
    analyze_tensor3(
        "depth confidence",
        refined.depth_confidence.as_ref(),
        reference.depth_confidence.as_ref(),
    );
    analyze_tensor4("aux (ray)", refined.aux.as_ref(), reference.aux.as_ref());
    analyze_tensor3(
        "aux confidence",
        refined.aux_confidence.as_ref(),
        reference.aux_confidence.as_ref(),
    );
    analyze_tensor3(
        "pose encoding",
        refined.pose_encoding.as_ref(),
        reference.pose_encoding.as_ref(),
    );
    analyze_tensor4(
        "extrinsics",
        refined.extrinsics.as_ref(),
        reference.extrinsics.as_ref(),
    );
    analyze_tensor4(
        "intrinsics",
        refined.intrinsics.as_ref(),
        reference.intrinsics.as_ref(),
    );
    analyze_backbone_tokens(&trace, &reference.backbone_tokens);
    analyze_aux_stage_necks(
        "inference aux_stage",
        trace.aux_stage_necks.as_ref(),
        &reference.aux_stage_necks,
    );
    analyze_tensor4(
        "aux logits",
        trace.aux_logits.as_ref(),
        reference.aux_logits.as_ref(),
    );
    analyze_tensor4(
        "aux head_input",
        trace.aux_head_input.as_ref(),
        reference.aux_head_input.as_ref(),
    );
    analyze_head_with_reference_tokens(&model, &device, &reference)?;

    Ok(())
}

struct PreparedInput {
    width: usize,
    height: usize,
    rgb: RgbImage,
}

fn load_prepared_image(path: &str, target: usize) -> Result<PreparedInput, String> {
    let rgb = image::open(path)
        .map_err(|err| format!("Failed to open image `{path}`: {err}"))?
        .to_rgb8();
    let prepared = prepare_depth_anything3_image(&rgb, target)?;
    Ok(PreparedInput {
        width: prepared.width,
        height: prepared.height,
        rgb: prepared.rgb,
    })
}

#[derive(Clone)]
struct ReferenceTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

struct ReferenceDa3 {
    metric_input: Option<ReferenceTensor>,
    depth: ReferenceTensor,
    depth_confidence: Option<ReferenceTensor>,
    aux: Option<ReferenceTensor>,
    aux_confidence: Option<ReferenceTensor>,
    pose_encoding: Option<ReferenceTensor>,
    extrinsics: Option<ReferenceTensor>,
    intrinsics: Option<ReferenceTensor>,
    backbone_tokens: Vec<ReferenceTensor>,
    aux_stage_necks: Vec<ReferenceTensor>,
    aux_logits: Option<ReferenceTensor>,
    aux_head_input: Option<ReferenceTensor>,
}

fn load_reference(path: &Path) -> Result<ReferenceDa3, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let depth = read_required(&tensors, "depth")?;
    Ok(ReferenceDa3 {
        metric_input: read_optional(&tensors, "metric_input"),
        depth,
        depth_confidence: read_optional(&tensors, "depth_confidence"),
        aux: read_optional(&tensors, "ray"),
        aux_confidence: read_optional(&tensors, "ray_confidence"),
        pose_encoding: read_optional(&tensors, "pose_encoding"),
        extrinsics: read_optional(&tensors, "extrinsics"),
        intrinsics: read_optional(&tensors, "intrinsics"),
        backbone_tokens: load_backbone_tokens(&tensors),
        aux_stage_necks: load_aux_stage_necks(&tensors),
        aux_logits: read_optional(&tensors, "aux_logits"),
        aux_head_input: read_optional(&tensors, "aux_head_input"),
    })
}

fn read_required(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<ReferenceTensor, Box<dyn std::error::Error>> {
    let view = tensors
        .tensor(name)
        .map_err(|err| format!("Reference tensor `{name}` missing: {err}"))?;
    Ok(ReferenceTensor {
        shape: view.shape().to_vec(),
        data: tensor_view_to_vec(&view),
    })
}

fn read_optional(tensors: &SafeTensors<'_>, name: &str) -> Option<ReferenceTensor> {
    tensors.tensor(name).ok().map(|view| ReferenceTensor {
        shape: view.shape().to_vec(),
        data: tensor_view_to_vec(&view),
    })
}

fn load_backbone_tokens(tensors: &SafeTensors<'_>) -> Vec<ReferenceTensor> {
    let mut result = Vec::new();
    for idx in 0usize.. {
        let key = format!("backbone_tokens.stage{idx}");
        match tensors.tensor(&key) {
            Ok(view) => result.push(ReferenceTensor {
                shape: view.shape().to_vec(),
                data: tensor_view_to_vec(&view),
            }),
            Err(_) => break,
        }
    }
    result
}

fn load_aux_stage_necks(tensors: &SafeTensors<'_>) -> Vec<ReferenceTensor> {
    let mut result = Vec::new();
    for idx in 0usize.. {
        let key = format!("aux_stage_necks.stage{idx}");
        match tensors.tensor(&key) {
            Ok(view) => result.push(ReferenceTensor {
                shape: view.shape().to_vec(),
                data: tensor_view_to_vec(&view),
            }),
            Err(_) => break,
        }
    }
    result
}

fn tensor_view_to_vec(view: &TensorView) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn convert_reference_tokens(
    tensors: &[ReferenceTensor],
    device: &NdArrayDevice,
) -> Result<Vec<Tensor<NdBackend, 3>>, String> {
    tensors
        .iter()
        .map(|tensor| tensor_from_reference::<3>(tensor, device))
        .collect()
}

fn reference_depth_hw(reference: &ReferenceDa3) -> Result<(usize, usize), String> {
    let shape = reference.depth.shape.as_slice();
    match shape {
        [_, h, w] => Ok((*h, *w)),
        [h, w] => Ok((*h, *w)),
        _ => Err(format!(
            "Depth reference shape {:?} is not supported",
            reference.depth.shape
        )),
    }
}

fn analyze_head_with_reference_tokens(
    model: &DepthAnything3<NdBackend>,
    device: &NdArrayDevice,
    reference: &ReferenceDa3,
) -> Result<(), String> {
    if reference.backbone_tokens.is_empty() {
        println!("Reference lacks backbone tokens; skipping head-only comparison.");
        return Ok(());
    }
    let tokens = convert_reference_tokens(&reference.backbone_tokens, device)?;
    let (height, width) = reference_depth_hw(reference)?;
    let (head_result, aux_stage_necks, aux_logits, aux_head_input) =
        model.infer_from_tokens(&tokens, height, width);
    println!("--- Head-only comparison using Torch backbone tokens ---");
    analyze_tensor3(
        "head depth",
        Some(&head_result.depth),
        Some(&reference.depth),
    );
    analyze_tensor3(
        "head depth_confidence",
        head_result.depth_confidence.as_ref(),
        reference.depth_confidence.as_ref(),
    );
    analyze_tensor4("head aux", head_result.aux.as_ref(), reference.aux.as_ref());
    analyze_tensor3(
        "head aux_confidence",
        head_result.aux_confidence.as_ref(),
        reference.aux_confidence.as_ref(),
    );
    analyze_aux_stage_necks(
        "head aux_stage",
        aux_stage_necks.as_ref(),
        &reference.aux_stage_necks,
    );
    analyze_tensor4(
        "head aux_logits",
        aux_logits.as_ref(),
        reference.aux_logits.as_ref(),
    );
    analyze_tensor4(
        "head aux_head_input",
        aux_head_input.as_ref(),
        reference.aux_head_input.as_ref(),
    );

    Ok(())
}

fn tensor_from_reference<const D: usize>(
    reference: &ReferenceTensor,
    device: &NdArrayDevice,
) -> Result<Tensor<NdBackend, D>, String> {
    if reference.shape.len() != D {
        return Err(format!(
            "reference tensor rank {} does not match expected {D}",
            reference.shape.len()
        ));
    }
    let dims = reshape_dims::<D>(&reference.shape)?;
    Ok(Tensor::<NdBackend, 1>::from_floats(reference.data.as_slice(), device).reshape(dims))
}

fn reshape_dims<const D: usize>(shape: &[usize]) -> Result<[i32; D], String> {
    if shape.len() != D {
        return Err(format!(
            "shape {:?} cannot be reshaped into {}-D tensor",
            shape, D
        ));
    }
    let mut dims = [0i32; D];
    for (idx, &dim) in shape.iter().enumerate() {
        dims[idx] = dim
            .try_into()
            .map_err(|_| format!("dimension {dim} exceeds i32::MAX"))?;
    }
    Ok(dims)
}

fn analyze_tensor3(
    label: &str,
    tensor: Option<&Tensor<NdBackend, 3>>,
    reference: Option<&ReferenceTensor>,
) {
    match (tensor, reference) {
        (Some(tensor), Some(reference)) => {
            let data = tensor.clone().into_data().convert::<f32>();
            let values = data
                .to_vec::<f32>()
                .expect("failed to read tensor values for comparison");
            compare_tensors(label, &values, reference);
        }
        (Some(_), None) => println!("{label}: missing Torch reference tensor"),
        (None, Some(_)) => println!("{label}: model did not emit this tensor"),
        (None, None) => println!("{label}: not available in either model or reference"),
    }
}

fn analyze_tensor4(
    label: &str,
    tensor: Option<&Tensor<NdBackend, 4>>,
    reference: Option<&ReferenceTensor>,
) {
    match (tensor, reference) {
        (Some(tensor), Some(reference)) => {
            let data = tensor.clone().into_data().convert::<f32>();
            let values = data
                .to_vec::<f32>()
                .expect("failed to read tensor values for comparison");
            compare_tensors(label, &values, reference);
        }
        (Some(_), None) => println!("{label}: missing Torch reference tensor"),
        (None, Some(_)) => println!("{label}: model did not emit this tensor"),
        (None, None) => println!("{label}: not available in either model or reference"),
    }
}

fn analyze_backbone_tokens(trace: &DepthTrace<NdBackend>, reference: &[ReferenceTensor]) {
    if reference.is_empty() {
        println!("Reference lacks backbone token traces; skipping backbone comparison.");
        return;
    }
    if trace.backbone_tokens.len() != reference.len() {
        println!(
            "Backbone token stage count mismatch: model={}, reference={}",
            trace.backbone_tokens.len(),
            reference.len()
        );
    }
    for idx in 0..reference.len() {
        let label = format!("backbone_tokens.stage{idx}");
        let Some(burn_tensor) = trace.backbone_tokens.get(idx) else {
            println!("{label}: model did not emit this stage");
            continue;
        };
        let (values, shape) = tensor3_to_vec(burn_tensor);
        if values.len() != reference[idx].data.len() {
            println!(
                "{label}: size mismatch model={} reference={} (model shape {:?}, reference shape {:?})",
                values.len(),
                reference[idx].data.len(),
                shape,
                reference[idx].shape
            );
            continue;
        }
        compare_tensors(&label, &values, &reference[idx]);
    }
}

fn compare_tensors(label: &str, burn: &[f32], reference: &ReferenceTensor) {
    if burn.len() != reference.data.len() {
        println!(
            "{label}: size mismatch model={} reference={} for shape {:?}",
            burn.len(),
            reference.data.len(),
            reference.shape
        );
        return;
    }
    let mut mae = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for (lhs, rhs) in burn.iter().zip(reference.data.iter()) {
        let diff = (lhs - rhs).abs();
        mae += diff as f64;
        if diff > max_abs {
            max_abs = diff;
        }
        if lhs.is_finite() {
            min_val = min_val.min(*lhs);
            max_val = max_val.max(*lhs);
        }
    }
    if !burn.is_empty() {
        mae /= burn.len() as f64;
    }
    println!(
        "{label}: shape {:?}, mae={mae:.6}, max_abs={max_abs:.6}, range=[{min_val:.4}, {max_val:.4}]",
        reference.shape
    );
}

fn tensor3_to_vec(tensor: &Tensor<NdBackend, 3>) -> (Vec<f32>, Vec<usize>) {
    let data = tensor.clone().into_data().convert::<f32>();
    let shape = data.shape.clone();
    let values = data
        .to_vec::<f32>()
        .expect("failed to read backbone tensor values");
    (values, shape)
}

fn tensor4_to_vec(tensor: &Tensor<NdBackend, 4>) -> (Vec<f32>, Vec<usize>) {
    let data = tensor.clone().into_data().convert::<f32>();
    let shape = data.shape.clone();
    let values = data
        .to_vec::<f32>()
        .expect("failed to read aux tensor values");
    (values, shape)
}

fn analyze_aux_stage_necks(
    label_prefix: &str,
    burn: Option<&Vec<Tensor<NdBackend, 4>>>,
    reference: &[ReferenceTensor],
) {
    if reference.is_empty() {
        println!("{label_prefix}: reference lacks aux stage tensors; skipping.");
        return;
    }
    match burn {
        Some(stages) => {
            for idx in 0..reference.len() {
                let label = format!("{label_prefix}.stage{idx}");
                let Some(stage) = stages.get(idx) else {
                    println!("{label}: model did not emit this stage");
                    continue;
                };
                let (values, shape) = tensor4_to_vec(stage);
                if values.len() != reference[idx].data.len() {
                    println!(
                        "{label}: size mismatch model={} reference={} (model shape {:?}, reference shape {:?})",
                        values.len(),
                        reference[idx].data.len(),
                        shape,
                        reference[idx].shape
                    );
                    continue;
                }
                compare_tensors(&label, &values, &reference[idx]);
            }
        }
        None => println!("{label_prefix}: model did not emit aux stage tensors"),
    }
}
