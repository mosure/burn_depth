use std::path::PathBuf;

use burn::{
    backend::NdArray,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::Backend,
};
use burn_depth::model::depth_anything3::{DepthAnything3, DepthAnything3Config};
use burn_store::{
    ApplyResult, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use clap::Parser;

type ImportBackend = NdArray<f32>;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Convert DA3 safetensors weights into Burn checkpoints"
)]
struct Args {
    #[arg(
        long,
        value_name = "PATH",
        default_value = "assets/model/da3_metric_large.safetensors"
    )]
    checkpoint: PathBuf,

    #[arg(
        long,
        value_name = "PATH",
        default_value = "assets/model/da3_metric_large.mpk"
    )]
    output: PathBuf,

    #[arg(long, value_name = "BOOL", default_value_t = false)]
    dry_run: bool,

    #[arg(long, value_name = "PATH")]
    dump_template: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if !args.checkpoint.exists() {
        return Err(format!("Checkpoint `{}` not found.", args.checkpoint.display()).into());
    }

    let device = <ImportBackend as Backend>::Device::default();
    let config = DepthAnything3Config::metric_large();
    let mut model = DepthAnything3::<ImportBackend>::new(&device, config);

    if let Some(path) = &args.dump_template {
        export_template(&model, path)?;
        if args.dry_run {
            return Ok(());
        }
    }

    let remapper = KeyRemapper::new()
        .add_pattern(r"^model\.", "")?
        .add_pattern(r"^(backbone\.pretrained\..*\.norm\d+)\.weight$", "$1.gamma")?
        .add_pattern(r"^(backbone\.pretrained\..*\.norm\d+)\.bias$", "$1.beta")?
        .add_pattern(r"^(backbone\.pretrained\.norm)\.weight$", "$1.gamma")?
        .add_pattern(r"^(backbone\.pretrained\.norm)\.bias$", "$1.beta")?
        .add_pattern(
            r"^(head\.resize_layers\.(0|1))\.weight$",
            "$1.conv_t.weight",
        )?
        .add_pattern(r"^(head\.resize_layers\.(0|1))\.bias$", "$1.conv_t.bias")?
        .add_pattern(r"^(head\.resize_layers\.3)\.weight$", "$1.conv.weight")?
        .add_pattern(r"^(head\.resize_layers\.3)\.bias$", "$1.conv.bias")?
        .add_pattern(
            r"^(head\.scratch\.output_conv2)\.0\.(weight|bias)$",
            "$1.conv1.$2",
        )?
        .add_pattern(
            r"^(head\.scratch\.output_conv2)\.2\.(weight|bias)$",
            "$1.conv2.$2",
        )?
        .add_pattern(
            r"^(head\.scratch\.refinenet\d+)\.resConfUnit1\.",
            "$1.residual1.",
        )?
        .add_pattern(
            r"^(head\.scratch\.refinenet\d+)\.resConfUnit2\.",
            "$1.residual2.",
        )?;
    let mut store = SafetensorsStore::from_file(&args.checkpoint);
    store = store
        .remap(remapper)
        .with_from_adapter(PyTorchToBurnAdapter::default())
        .allow_partial(true);

    println!("Loading {}", args.checkpoint.display());
    let result = model
        .load_from(&mut store)
        .map_err(|err| format!("Failed to apply checkpoint: {err}"))?;
    report_result(&result);

    if args.dry_run {
        println!("Dry run enabled; checkpoint not written.");
        return Ok(());
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(args.output.clone(), &recorder)
        .map_err(|err| format!("Failed to save checkpoint: {err}"))?;
    println!("Saved Burn checkpoint to {}", args.output.display());
    Ok(())
}

fn report_result(result: &ApplyResult) {
    println!(
        "Applied {} tensors ({} skipped, {} missing, {} unused).",
        result.applied.len(),
        result.skipped.len(),
        result.missing.len(),
        result.unused.len()
    );
    if !result.missing.is_empty() {
        for key in &result.missing {
            println!("Missing tensor: {key}");
        }
    }
    if !result.unused.is_empty() {
        for key in &result.unused {
            println!("Unused tensor: {key}");
        }
    }
}

fn export_template<B: Backend>(
    model: &DepthAnything3<B>,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut paths: Vec<String> = model
        .clone()
        .collect(None, None)
        .into_iter()
        .map(|snapshot| snapshot.full_path())
        .collect();
    paths.sort();
    std::fs::write(path, paths.join("\n"))?;
    println!("Wrote template paths to {}", path.display());
    Ok(())
}
