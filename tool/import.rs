use std::path::PathBuf;

use burn::{
    module::Module,
    record::serde::{
        data::{NestedValue, remap, unflatten},
        de::Deserializer,
        ser::Serializer,
    },
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Record, Recorder},
};
use burn_depth_pro::model::depth_pro::{DepthPro, DepthProConfig};
use burn_import::common::{
    adapter::PyTorchAdapter,
    tensor_snapshot::{TensorSnapshotWrapper, print_debug_info},
};
use burn_store::pytorch::PytorchReader;
use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

type Backend = burn_ndarray::NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let model = DepthPro::<Backend>::new(&device, DepthProConfig::default());

    if std::env::var("EXPORT_TEMPLATE").is_ok() {
        dump_template(model.clone())?;
        return Ok(());
    }

    let default_record = model.clone().into_record();
    let record = load_torch_checkpoint(&device, &default_record)?;
    let model = model.load_record(record);

    let output_path = PathBuf::from("assets/model/depth_pro.mpk");
    model.clone().save_file(
        output_path.clone(),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
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
    let item = record.into_item::<FullPrecisionSettings>();
    let mut value = serde_json::to_value(&item)?;
    prune_bytes(&mut value);

    let mut paths = Vec::new();
    collect_paths(&value, String::new(), &mut paths);
    std::fs::write(
        "assets/model/depth_pro_template_paths.txt",
        paths.join("\n"),
    )?;
    println!("Wrote template paths to assets/model/depth_pro_template_paths.txt");
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

fn load_torch_checkpoint(
    device: &<Backend as burn::tensor::backend::Backend>::Device,
    default_record: &<DepthPro<Backend> as Module<Backend>>::Record,
) -> Result<<DepthPro<Backend> as Module<Backend>>::Record, Box<dyn std::error::Error>> {
    let path = PathBuf::from("assets/model/depth_pro.pt");

    let reader = PytorchReader::new(&path)?;
    let tensors: HashMap<String, TensorSnapshotWrapper> = reader
        .into_tensors()
        .into_iter()
        .map(|(key, snapshot)| (key, TensorSnapshotWrapper(snapshot)))
        .collect();

    let remap_rules = key_remap_rules()?;
    let (tensors, remapped_keys) = remap(tensors, remap_rules);

    if std::env::var("IMPORT_DEBUG").is_ok() {
        print_debug_info(&tensors, remapped_keys);
    }

    let mut nested = unflatten::<FullPrecisionSettings, _>(tensors)?;
    ensure_mask_tokens(&mut nested, default_record)?;

    let deserializer =
        Deserializer::<PyTorchAdapter<FullPrecisionSettings, Backend>>::new(nested, true);
    let record = <DepthPro<Backend> as Module<Backend>>::Record::deserialize(deserializer)?;

    println!("Successfully loaded PyTorch checkpoint.");
    Ok(record)
}
fn key_remap_rules() -> Result<Vec<(Regex, String)>, regex::Error> {
    Ok(vec![
        (
            Regex::new(r"^decoder\.convs\.(\d+)\.(weight|bias)$")?,
            "decoder.convs.$1.conv.$2".into(),
        ),
        (
            Regex::new(r"^decoder\.fusions\.(\d+)\.resnet([12])\.residual\.1\.(weight|bias)$")?,
            "decoder.fusions.$1.resnet$2.conv1.$3".into(),
        ),
        (
            Regex::new(r"^decoder\.fusions\.(\d+)\.resnet([12])\.residual\.3\.(weight|bias)$")?,
            "decoder.fusions.$1.resnet$2.conv2.$3".into(),
        ),
        (
            Regex::new(r"^head\.0\.(weight|bias)$")?,
            "head.conv0.$1".into(),
        ),
        (
            Regex::new(r"^head\.1\.(weight|bias)$")?,
            "head.deconv.$1".into(),
        ),
        (
            Regex::new(r"^head\.2\.(weight|bias)$")?,
            "head.conv1.$1".into(),
        ),
        (
            Regex::new(r"^head\.4\.(weight|bias)$")?,
            "head.conv_out.$1".into(),
        ),
        (
            Regex::new(r"^fov\.head\.(\d+)\.(weight|bias)$")?,
            "fov.head_blocks.$1.conv.$2".into(),
        ),
    ])
}

fn serialize_to_nested<T: Serialize>(value: &T) -> Result<NestedValue, Box<dyn std::error::Error>> {
    Ok(value.serialize(Serializer::new())?)
}

fn ensure_mask_tokens(
    nested: &mut NestedValue,
    default_record: &<DepthPro<Backend> as Module<Backend>>::Record,
) -> Result<(), Box<dyn std::error::Error>> {
    insert_nested(
        nested,
        &["encoder", "patch_encoder", "mask_token"],
        serialize_to_nested(&default_record.encoder.patch_encoder.mask_token)?,
    );

    insert_nested(
        nested,
        &["encoder", "image_encoder", "mask_token"],
        serialize_to_nested(&default_record.encoder.image_encoder.mask_token)?,
    );

    if let Some(fov_record) = default_record.fov.as_ref() {
        insert_nested(
            nested,
            &["fov", "encoder", "mask_token"],
            serialize_to_nested(&fov_record.encoder.mask_token)?,
        );
    }

    Ok(())
}

fn insert_nested(target: &mut NestedValue, path: &[&str], value: NestedValue) {
    match target {
        NestedValue::Map(map) => {
            if path.is_empty() {
                *target = value;
            } else {
                let entry = map
                    .entry(path[0].to_string())
                    .or_insert_with(|| NestedValue::Map(HashMap::new()));
                if path.len() == 1 {
                    *entry = value;
                } else {
                    insert_nested(entry, &path[1..], value.clone());
                }
            }
        }
        _ => {
            *target = NestedValue::Map(HashMap::new());
            insert_nested(target, path, value);
        }
    }
}
