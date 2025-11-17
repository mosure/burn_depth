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
use clap::{Parser, ValueEnum};

type ImportBackend = NdArray<f32>;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Convert DA3 safetensors weights into Burn checkpoints"
)]
struct Args {
    #[arg(long, value_enum, default_value_t = ModelVariant::MetricLarge)]
    variant: ModelVariant,

    #[arg(long, value_name = "PATH")]
    checkpoint: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    #[arg(long, value_name = "BOOL", default_value_t = false)]
    dry_run: bool,

    #[arg(long, value_name = "PATH")]
    dump_template: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let checkpoint = args
        .checkpoint
        .unwrap_or_else(|| args.variant.default_checkpoint());
    let output = args.output.unwrap_or_else(|| args.variant.default_output());

    if !checkpoint.exists() {
        return Err(format!("Checkpoint `{}` not found.", checkpoint.display()).into());
    }

    let device = <ImportBackend as Backend>::Device::default();
    let config = args.variant.config();
    let mut model = DepthAnything3::<ImportBackend>::new(&device, config.clone());
    let head_prefix = if config.head.dual_head {
        "head_dual"
    } else {
        "head_mono"
    };

    if let Some(path) = &args.dump_template {
        export_template(&model, path)?;
        if args.dry_run {
            return Ok(());
        }
    }

    let remapper = KeyRemapper::new()
        .add_pattern(r"^model\.", "")?
        .add_pattern(r"^head\.", format!("{head_prefix}."))?
        .add_pattern(
            r"^cam_dec\.backbone\.0\.(weight|bias)$",
            "camera_decoder.backbone_1.$1",
        )?
        .add_pattern(
            r"^cam_dec\.backbone\.2\.(weight|bias)$",
            "camera_decoder.backbone_2.$1",
        )?
        .add_pattern(r"^cam_dec\.fc_t\.(weight|bias)$", "camera_decoder.fc_t.$1")?
        .add_pattern(
            r"^cam_dec\.fc_qvec\.(weight|bias)$",
            "camera_decoder.fc_qvec.$1",
        )?
        .add_pattern(
            r"^cam_dec\.fc_fov\.0\.(weight|bias)$",
            "camera_decoder.fc_fov.$1",
        )?
        .add_pattern(r"^cam_dec\.", "camera_decoder.")?
        .add_pattern(r"^cam_enc\.", "camera_encoder.")?
        .add_pattern(r"^(backbone\.pretrained\..*\.norm\d+)\.weight$", "$1.gamma")?
        .add_pattern(r"^(backbone\.pretrained\..*\.norm\d+)\.bias$", "$1.beta")?
        .add_pattern(r"^(backbone\.pretrained\.norm)\.weight$", "$1.gamma")?
        .add_pattern(r"^(backbone\.pretrained\.norm)\.bias$", "$1.beta")?
        .add_pattern(
            r"^(backbone\.pretrained\..*\.attn\.q_norm)\.weight$",
            "$1.gamma",
        )?
        .add_pattern(
            r"^(backbone\.pretrained\..*\.attn\.q_norm)\.bias$",
            "$1.beta",
        )?
        .add_pattern(
            r"^(backbone\.pretrained\..*\.attn\.k_norm)\.weight$",
            "$1.gamma",
        )?
        .add_pattern(
            r"^(backbone\.pretrained\..*\.attn\.k_norm)\.bias$",
            "$1.beta",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\..*norm\d*)\.weight$"),
            "$1.gamma",
        )?
        .add_pattern(&format!(r"^({head_prefix}\..*norm\d*)\.bias$"), "$1.beta")?
        .add_pattern(
            &format!(r"^({head_prefix}\.resize_layers\.(0|1))\.weight$"),
            "$1.conv_t.weight",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.resize_layers\.(0|1))\.bias$"),
            "$1.conv_t.bias",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.resize_layers\.3)\.weight$"),
            "$1.conv.weight",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.resize_layers\.3)\.bias$"),
            "$1.conv.bias",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.output_conv2)\.0\.(weight|bias)$"),
            "$1.conv1.$2",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.output_conv2)\.2\.(weight|bias)$"),
            "$1.conv2.$2",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.refinenet\d+)\.resConfUnit1\."),
            "$1.residual1.",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.refinenet\d+)\.resConfUnit2\."),
            "$1.residual2.",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.refinenet\d+_aux)\.resConfUnit1\."),
            "$1.residual1.",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.refinenet\d+_aux)\.resConfUnit2\."),
            "$1.residual2.",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.output_conv1_aux\.\d+)\.(\d+)\.(weight|bias)$"),
            "$1.layers.$2.$3",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.output_conv2_aux\.\d+)\.0\.(weight|bias)$"),
            "$1.reduce.$2",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.output_conv2_aux\.\d+)\.2\.(weight|bias)$"),
            "$1.norm.layer_norm.$2",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.output_conv2_aux\.\d+)\.5\.(weight|bias)$"),
            "$1.project.$2",
        )?
        .add_pattern(
            &format!(
                r"^({head_prefix}\.scratch\.output_conv2_aux\.\d+\.norm\.layer_norm)\.weight$"
            ),
            "$1.gamma",
        )?
        .add_pattern(
            &format!(r"^({head_prefix}\.scratch\.output_conv2_aux\.\d+\.norm\.layer_norm)\.bias$"),
            "$1.beta",
        )?
        .add_pattern(r"^(camera_encoder\..*norm\d+)\.weight$", "$1.gamma")?
        .add_pattern(r"^(camera_encoder\..*norm\d+)\.bias$", "$1.beta")?
        .add_pattern(r"^(camera_encoder\..*norm)\.weight$", "$1.gamma")?
        .add_pattern(r"^(camera_encoder\..*norm)\.bias$", "$1.beta")?
        .add_pattern(
            r"^(camera_encoder\.pose_branch\.fc1)\.(weight|bias)$",
            "$1.$2",
        )?
        .add_pattern(
            r"^(camera_encoder\.pose_branch\.fc2)\.(weight|bias)$",
            "$1.$2",
        )?;
    let mut store = SafetensorsStore::from_file(&checkpoint);
    store = store
        .remap(remapper)
        .with_from_adapter(PyTorchToBurnAdapter::default())
        .allow_partial(true);

    println!("Loading {}", checkpoint.display());
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
        .save_file(output.clone(), &recorder)
        .map_err(|err| format!("Failed to save checkpoint: {err}"))?;
    println!("Saved Burn checkpoint to {}", output.display());
    Ok(())
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ModelVariant {
    MetricLarge,
    Small,
}

impl ModelVariant {
    fn config(self) -> DepthAnything3Config {
        match self {
            ModelVariant::MetricLarge => DepthAnything3Config::metric_large(),
            ModelVariant::Small => DepthAnything3Config::metric_small(),
        }
    }

    fn default_checkpoint(self) -> PathBuf {
        match self {
            ModelVariant::MetricLarge => PathBuf::from("assets/model/da3_metric_large.safetensors"),
            ModelVariant::Small => PathBuf::from("assets/model/da3_small.safetensors"),
        }
    }

    fn default_output(self) -> PathBuf {
        match self {
            ModelVariant::MetricLarge => PathBuf::from("assets/model/da3_metric_large.mpk"),
            ModelVariant::Small => PathBuf::from("assets/model/da3_small.mpk"),
        }
    }
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
