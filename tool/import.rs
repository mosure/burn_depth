use bevy_args::{
    parse_args,
    Deserialize,
    Parser,
    Serialize,
};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

use burn_depth_pro::model::depth_pro::DepthProRecord;


#[derive(
    Clone,
    Debug,
    Default,
    Serialize,
    Deserialize,
    Parser,
)]
#[command(about = "burn_depth_pro import", version, long_about = None)]
pub struct DepthProImportConfig {
    #[arg(long, default_value = "./assets/models/depth_pro.pth")]
    pub weights_path: String,

    #[arg(long, default_value = "./assets/models/depth_pro")]
    pub output_path: String,
}


type Backend = burn::backend::NdArray<f32>;

fn main() {
    let args = parse_args::<DepthProImportConfig>();

    let device = Default::default();

    println!("loading weights from: {}", args.weights_path);

    let load_args = LoadArgs::new(args.weights_path.into())
        .with_debug_print();
    
    let record: DepthProRecord<Backend> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("failed to decode state");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, args.output_path.into())
        .expect("failed to save model record");

    println!("Model successfully imported!");
}
