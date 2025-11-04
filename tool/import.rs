use std::{
    collections::HashSet,
    convert::TryInto,
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Record},
};
use burn_depth::model::depth_pro::{DepthPro, DepthProConfig};
use burn_store::{
    ApplyResult, ModuleSnapshot, TensorSnapshot,
    pytorch::{PytorchReader, PytorchStore},
};
use serde_json::Value;

type Backend = burn::backend::NdArray<f32>;

const CHECKPOINT_PATH: &str = "assets/model/depth_pro.pt";
const OUTPUT_PATH: &str = "assets/model/depth_pro.mpk";
const TEMPLATE_PATH: &str = "assets/model/depth_pro_template_paths.txt";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let mut model = DepthPro::<Backend>::new(&device, DepthProConfig::default());

    if std::env::var("EXPORT_TEMPLATE").is_ok() {
        dump_template(model.clone())?;
        return Ok(());
    }

    let debug = std::env::var("IMPORT_DEBUG").is_ok();
    let checkpoint_path = PathBuf::from(CHECKPOINT_PATH);
    if !checkpoint_path.exists() {
        return Err(format!(
            "PyTorch checkpoint `{}` not found.",
            checkpoint_path.display()
        )
        .into());
    }
    if debug {
        debug_print_checkpoint(&checkpoint_path)?;
        debug_print_model_paths(&model);
    }
    let mut store = build_store(&checkpoint_path);

    println!("Loading checkpoint {}", checkpoint_path.display());
    let result = model
        .load_from(&mut store)
        .map_err(|err| format!("Failed to apply PyTorch checkpoint: {err}"))?;

    report_apply_result(&result, debug)?;
    model.fix_conv_transpose_weights();

    if std::env::var("IMPORT_VALIDATE").is_ok() {
        validate_against_reference(&model, &checkpoint_path)?;
    }

    let output_path = PathBuf::from(OUTPUT_PATH);
    model.clone().save_file(
        output_path.clone(),
        &NamedMpkFileRecorder::<HalfPrecisionSettings>::new(),
    )?;

    println!("Saved Burn checkpoint to {}", output_path.display());
    Ok(())
}

fn dump_template(model: DepthPro<Backend>) -> Result<(), Box<dyn std::error::Error>> {
    fn print_type_of<T>(_value: &T, label: &str) {
        println!("{}: {}", label, std::any::type_name::<T>());
    }

    let record = model.into_record();
    print_type_of(&record, "Record type");
    print_type_of(&record.encoder, "Encoder record type");
    print_type_of(&record.encoder.patch_encoder, "Patch encoder record type");
    let mut value = serde_json::to_value(&record.into_item::<HalfPrecisionSettings>())?;
    prune_bytes(&mut value);

    let mut paths = Vec::new();
    collect_paths(&value, String::new(), &mut paths);
    std::fs::write(TEMPLATE_PATH, paths.join("\n"))?;
    println!("Wrote template paths to {}", TEMPLATE_PATH);
    Ok(())
}

fn prune_bytes(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(entry) = map.get_mut("bytes") {
                *entry = Value::Null;
            }
            for child in map.values_mut() {
                prune_bytes(child);
            }
        }
        Value::Array(arr) => {
            for child in arr {
                prune_bytes(child);
            }
        }
        _ => {}
    }
}

fn collect_paths(value: &Value, prefix: String, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            if map.contains_key("bytes") {
                out.push(format!("{prefix}bytes"));
            }
            for (key, child) in map {
                let new_prefix = if prefix.is_empty() {
                    format!("{}.", key)
                } else {
                    format!("{}{}.", prefix, key)
                };
                collect_paths(child, new_prefix, out);
            }
        }
        Value::Array(arr) => {
            for (index, child) in arr.iter().enumerate() {
                let new_prefix = format!("{}{}.", prefix, index);
                collect_paths(child, new_prefix, out);
            }
        }
        _ => {}
    }
}

fn build_store(path: &Path) -> PytorchStore {
    let mut store = PytorchStore::from_file(path);
    for &(from, to) in key_remap_rules() {
        store = store.with_key_remapping(from, to);
    }
    store.allow_partial(true).validate(true)
}

fn validate_against_reference(
    model: &DepthPro<Backend>,
    checkpoint_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use safetensors::tensor::SafeTensors;

    use burn::tensor::Tensor;

    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let reference_bytes = std::fs::read("assets/image/test.safetensors")?;
    let tensors = SafeTensors::deserialize(&reference_bytes)?;
    let load_tensor = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let view = tensors
            .tensor(name)
            .map_err(|_| format!("reference tensor `{}` missing", name))?;
        Ok(view
            .data()
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(bytes)
            })
            .collect())
    };

    let network_input = load_tensor("network_input")?;
    let network_shape: [usize; 4] = tensors
        .tensor("network_input")
        .map_err(|_| "reference missing `network_input` tensor")?
        .shape()
        .try_into()
        .map_err(|_| "invalid network input shape")?;
    let feature_input = Tensor::<Backend, 1>::from_floats(network_input.as_slice(), &device)
        .reshape([
            network_shape[0] as i32,
            network_shape[1] as i32,
            network_shape[2] as i32,
            network_shape[3] as i32,
        ]);

    let (canonical, decoder_feature, decoder_lowres, _fusion_outputs, _) =
        model.forward_with_decoder(feature_input.clone());

    let canonical_data = canonical
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("validate: canonical extraction failed: {err:?}"))?;
    let decoder_data = decoder_feature
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("validate: decoder extraction failed: {err:?}"))?;
    let decoder_lowres_data = decoder_lowres
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("validate: decoder lowres extraction failed: {err:?}"))?;

    let canonical_ref = load_tensor("canonical_inverse_depth")?;
    let decoder_ref = load_tensor("decoder_feature")?;
    let decoder_lowres_ref = load_tensor("decoder_lowres_feature")?;

    fn stats(prefix: &str, burn: &[f32], reference: &[f32]) {
        let mut sum_abs = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;

        for (&b, &r) in burn.iter().zip(reference.iter()) {
            let diff = b - r;
            let abs = diff.abs();
            sum_abs += abs;
            max_abs = max_abs.max(abs);
            let rel = if r.abs() > 1e-6 { abs / r.abs() } else { 0.0 };
            max_rel = max_rel.max(rel);
        }
        let mean_abs = sum_abs / burn.len() as f32;
        println!(
            "[IMPORT_VALIDATE] {prefix}: mean abs={mean_abs:.6}, max abs={max_abs:.6}, max rel={max_rel:.6}"
        );
    }

    stats("canonical", &canonical_data, &canonical_ref);
    stats("decoder", &decoder_data, &decoder_ref);
    stats("decoder_lowres", &decoder_lowres_data, &decoder_lowres_ref);

    let record = model.clone().into_record();
    println!(
        "[IMPORT_VALIDATE] record encoder.upsample_lowres.weight type: {}",
        std::any::type_name_of_val(&record.encoder.upsample_lowres.weight)
    );

    let mut collector = burn_store::Collector::new(None, None);
    model.visit(&mut collector);
    let tensors = collector.into_tensors();
    println!("[IMPORT_VALIDATE] Sample decoder.convs tensors:");
    for snapshot in tensors
        .iter()
        .filter(|tensor| tensor.full_path().starts_with("decoder.convs"))
        .take(4)
    {
        println!("  {}", snapshot.full_path());
    }
    println!("[IMPORT_VALIDATE] Sample decoder.fusions tensors:");
    for snapshot in tensors
        .iter()
        .filter(|tensor| tensor.full_path().starts_with("decoder.fusions"))
        .take(4)
    {
        println!("  {}", snapshot.full_path());
    }
    println!("[IMPORT_VALIDATE] decoder.fusions deconv tensors:");
    for snapshot in tensors
        .iter()
        .filter(|tensor| {
            let path = tensor.full_path();
            path.contains(".deconv.") || path.contains("upsample_lowres")
        })
        .take(6)
    {
        println!("  {}", snapshot.full_path());
    }

    let pytorch_reader = PytorchReader::new(checkpoint_path)?;
    let pytorch_tensors = pytorch_reader.into_tensors();
    let weight_checks = [
        (
            "encoder.upsample_lowres.weight",
            "encoder.upsample_lowres.weight",
        ),
        (
            "decoder.fusions.1.deconv.weight",
            "decoder.fusions.1.deconv.weight",
        ),
        (
            "decoder.fusions.2.deconv.weight",
            "decoder.fusions.2.deconv.weight",
        ),
        ("head.deconv.weight", "head.1.weight"),
        ("head.conv1.weight", "head.2.weight"),
        ("head.conv_out.weight", "head.4.weight"),
        ("head.deconv.bias", "head.1.bias"),
        ("head.conv1.bias", "head.2.bias"),
        ("head.conv_out.bias", "head.4.bias"),
    ];
    for (burn_key, torch_key) in weight_checks {
        if let Err(err) = report_weight_diff(&tensors, &pytorch_tensors, burn_key, torch_key) {
            println!("[IMPORT_VALIDATE] {burn_key} diff error: {err}");
        }
    }

    println!("[IMPORT_VALIDATE] Torch head tensor keys:");
    for key in pytorch_tensors
        .keys()
        .filter(|key| key.starts_with("head."))
        .take(8)
    {
        println!("  {key}");
    }

    Ok(())
}

fn report_weight_diff(
    burn_tensors: &[TensorSnapshot],
    torch_tensors: &std::collections::HashMap<String, TensorSnapshot>,
    burn_key: &str,
    torch_key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let burn_snapshot = burn_tensors
        .iter()
        .find(|tensor: &&TensorSnapshot| tensor.full_path() == burn_key)
        .ok_or_else(|| format!("missing burn tensor {burn_key}"))?;
    let torch_snapshot = torch_tensors
        .get(torch_key)
        .ok_or_else(|| format!("missing torch tensor {torch_key}"))?;

    let burn_data = burn_snapshot
        .to_data()?
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("burn tensor convert failed: {err:?}"))?;
    let torch_data = torch_snapshot
        .to_data()?
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("torch tensor convert failed: {err:?}"))?;

    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    for (&b, &t) in burn_data.iter().zip(torch_data.iter()) {
        let diff = b - t;
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
    }
    let mean_abs = sum_abs / burn_data.len() as f32;
    println!(
        "[IMPORT_VALIDATE] {burn_key} vs {torch_key} weight diff: mean abs={mean_abs:.6}, max abs={max_abs:.6}"
    );
    Ok(())
}

fn key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (
            r"^(encoder\.(?:patch_encoder|image_encoder)(?:\.blocks\.\d+)?\.norm\d?)\.weight$",
            "$1.gamma",
        ),
        (
            r"^(encoder\.(?:patch_encoder|image_encoder)(?:\.blocks\.\d+)?\.norm\d?)\.bias$",
            "$1.beta",
        ),
        (
            r"^(fov\.encoder(?:\.0)?(?:\.blocks\.\d+)?\.norm\d?)\.weight$",
            "$1.gamma",
        ),
        (
            r"^(fov\.encoder(?:\.0)?(?:\.blocks\.\d+)?\.norm\d?)\.bias$",
            "$1.beta",
        ),
        (
            r"^encoder\.upsample([0-2])\.0\.(weight|bias)$",
            "encoder.upsample$1.projection.$2",
        ),
        (
            r"^encoder\.upsample([0-2])\.1\.(weight|bias)$",
            "encoder.upsample$1.upsample.0.$2",
        ),
        (
            r"^encoder\.upsample_latent([0-1])\.0\.(weight|bias)$",
            "encoder.upsample_latent$1.projection.$2",
        ),
        (
            r"^encoder\.upsample_latent([0-1])\.1\.(weight|bias)$",
            "encoder.upsample_latent$1.upsample.0.$2",
        ),
        (
            r"^encoder\.upsample_latent([0-1])\.2\.(weight|bias)$",
            "encoder.upsample_latent$1.upsample.1.$2",
        ),
        (
            r"^encoder\.upsample_latent([0-1])\.3\.(weight|bias)$",
            "encoder.upsample_latent$1.upsample.2.$2",
        ),
        (
            r"^encoder\.upsample_lowres\.(weight|bias)$",
            "encoder.upsample_lowres.$1",
        ),
        (
            r"^encoder\.fuse_lowres\.(weight|bias)$",
            "encoder.fuse_lowres.$1",
        ),
        (
            r"^fov\.downsample\.(\d+)\.(weight|bias)$",
            "fov.downsample_blocks.$1.conv.$2",
        ),
        (
            r"^decoder\.convs\.(\d+)\.(weight|bias)$",
            "decoder.convs.$1.conv.$2",
        ),
        (
            r"^decoder\.fusions\.(\d+)\.resnet([12])\.residual\.1\.(weight|bias)$",
            "decoder.fusions.$1.resnet$2.conv1.$3",
        ),
        (
            r"^decoder\.fusions\.(\d+)\.resnet([12])\.residual\.3\.(weight|bias)$",
            "decoder.fusions.$1.resnet$2.conv2.$3",
        ),
        (
            r"^decoder\.fusions\.(\d+)\.deconv\.(weight|bias)$",
            "decoder.fusions.$1.deconv.$2",
        ),
        (
            r"^decoder\.fusions\.(\d+)\.out_conv\.(weight|bias)$",
            "decoder.fusions.$1.out_conv.$2",
        ),
        (r"^fov\.encoder\.0\.", "fov.encoder."),
        (r"^fov\.encoder\.1\.(weight|bias)$", "fov.encoder_proj.$1"),
        (r"^head\.0\.(weight|bias)$", "head.conv0.$1"),
        (r"^head\.1\.(weight|bias)$", "head.deconv.$1"),
        (r"^head\.2\.(weight|bias)$", "head.conv1.$1"),
        (r"^head\.4\.(weight|bias)$", "head.conv_out.$1"),
        (
            r"^fov\.head\.0\.(weight|bias)$",
            "fov.head_blocks.0.conv.$1",
        ),
        (
            r"^fov\.head\.2\.(weight|bias)$",
            "fov.head_blocks.1.conv.$1",
        ),
        (
            r"^fov\.head\.4\.(weight|bias)$",
            "fov.head_blocks.2.conv.$1",
        ),
    ]
}

fn allowed_missing() -> &'static [&'static str] {
    &[
        "encoder.patch_encoder.mask_token",
        "encoder.image_encoder.mask_token",
        "fov.encoder.mask_token",
    ]
}

fn report_apply_result(
    result: &ApplyResult,
    debug: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !result.errors.is_empty() {
        println!(
            "Encountered {} error(s) while applying tensors:",
            result.errors.len()
        );
        for error in &result.errors {
            println!("  - {}", error);
        }
        return Err("Failed to import checkpoint; see errors above.".into());
    }

    let allowed: HashSet<&str> = allowed_missing().iter().copied().collect();
    let unexpected: Vec<&String> = result
        .missing
        .iter()
        .filter(|key| !allowed.contains(key.as_str()))
        .collect();

    if !unexpected.is_empty() {
        println!(
            "Missing tensors not covered by the importer allowlist ({}):",
            unexpected.len()
        );
        for key in &unexpected {
            println!("  - {}", key);
        }
        return Err("Unexpected missing tensors encountered while importing.".into());
    }

    if !result.missing.is_empty() {
        println!(
            "Warning: {} tensor(s) absent from checkpoint; default initialization retained.",
            result.missing.len()
        );
        for key in &result.missing {
            println!("  - {}", key);
        }
    }

    if !result.unused.is_empty() {
        println!(
            "Warning: {} tensor(s) from checkpoint were unused.",
            result.unused.len()
        );
        if debug {
            for key in &result.unused {
                println!("  - {}", key);
            }
        }
    }

    if debug && !result.skipped.is_empty() {
        println!("Skipped {} tensor(s) due to filters:", result.skipped.len());
        for key in &result.skipped {
            println!("  - {}", key);
        }
    }

    println!(
        "Applied {} tensors ({} skipped, {} missing, {} unused).",
        result.applied.len(),
        result.skipped.len(),
        result.missing.len(),
        result.unused.len()
    );

    Ok(())
}

fn debug_print_model_paths(model: &DepthPro<Backend>) {
    let mut paths: Vec<String> = model
        .collect(None, None)
        .into_iter()
        .map(|snapshot| snapshot.full_path())
        .filter(|path| path.starts_with("fov"))
        .collect();

    paths.sort();
    if paths.is_empty() {
        println!("Debug: model has no `fov` tensors.");
    } else {
        println!("Debug: model tensor paths starting with `fov`:");
        for path in paths {
            println!("  {path}");
        }
    }
}

fn debug_print_checkpoint(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let tensors = PytorchReader::new(path)?.into_tensors();

    let mut head_entries: Vec<_> = tensors
        .iter()
        .filter(|(key, _)| key.starts_with("fov.head."))
        .map(|(key, snapshot)| (key.clone(), snapshot.shape.clone()))
        .collect();
    head_entries.sort_by(|a, b| a.0.cmp(&b.0));

    if head_entries.is_empty() {
        println!("Debug: no `fov.head` tensors found in checkpoint.");
    } else {
        println!("Debug: `fov.head` tensor shapes:");
        for (key, shape) in &head_entries {
            println!("  {key}: {:?}", shape);
        }
    }

    let mut patch_norm: Vec<_> = tensors
        .iter()
        .filter(|(key, _)| key.starts_with("encoder.patch_encoder") && key.contains(".norm"))
        .map(|(key, snapshot)| (key.clone(), snapshot.shape.clone()))
        .collect();
    patch_norm.sort_by(|a, b| a.0.cmp(&b.0));
    if !patch_norm.is_empty() {
        println!("Debug: sample `encoder.patch_encoder` gamma/beta tensors (first 10):");
        for (key, shape) in patch_norm.iter().take(10) {
            println!("  {key}: {:?}", shape);
        }
    }

    let mut image_norm: Vec<_> = tensors
        .iter()
        .filter(|(key, _)| key.starts_with("encoder.image_encoder") && key.contains(".norm"))
        .map(|(key, snapshot)| (key.clone(), snapshot.shape.clone()))
        .collect();
    image_norm.sort_by(|a, b| a.0.cmp(&b.0));
    if !image_norm.is_empty() {
        println!("Debug: sample `encoder.image_encoder` gamma/beta tensors (first 10):");
        for (key, shape) in image_norm.iter().take(10) {
            println!("  {key}: {:?}", shape);
        }
    }

    let mut fov_norm: Vec<_> = tensors
        .iter()
        .filter(|(key, _)| key.starts_with("fov.encoder") && key.contains(".norm"))
        .map(|(key, snapshot)| (key.clone(), snapshot.shape.clone()))
        .collect();
    fov_norm.sort_by(|a, b| a.0.cmp(&b.0));
    if !fov_norm.is_empty() {
        println!("Debug: sample `fov.encoder` norm tensors (first 10):");
        for (key, shape) in fov_norm.iter().take(10) {
            println!("  {key}: {:?}", shape);
        }
    }

    let mut upsample_entries: Vec<_> = tensors
        .iter()
        .filter(|(key, _)| key.starts_with("encoder.upsample"))
        .map(|(key, snapshot)| (key.clone(), snapshot.shape.clone()))
        .collect();
    upsample_entries.sort_by(|a, b| a.0.cmp(&b.0));
    if !upsample_entries.is_empty() {
        println!("Debug: sample `encoder.upsample` tensors (first 10):");
        for (key, shape) in upsample_entries.iter().take(10) {
            println!("  {key}: {:?}", shape);
        }
    }

    let mut fov_downsample: Vec<_> = tensors
        .iter()
        .filter(|(key, _)| key.starts_with("fov.downsample"))
        .map(|(key, snapshot)| (key.clone(), snapshot.shape.clone()))
        .collect();
    fov_downsample.sort_by(|a, b| a.0.cmp(&b.0));
    if !fov_downsample.is_empty() {
        println!("Debug: sample `fov.downsample` tensors:");
        for (key, shape) in fov_downsample.iter().take(10) {
            println!("  {key}: {:?}", shape);
        }
    }

    Ok(())
}
