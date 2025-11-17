#![recursion_limit = "512"]

use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn_depth::model::depth_anything3::{
    with_model_load_stack, DepthAnything3, DepthAnything3Config,
};

type WgpuBackend = burn::backend::Wgpu<f32>;

fn main() {
    let device = <WgpuBackend as Backend>::Device::default();
    let config = DepthAnything3Config::metric_small();
    println!("constructing Depth Anything 3 (wgpu)...");
    println!("loading checkpoint...");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let checkpoint = "assets/model/da3_small.mpk";
    with_model_load_stack(|| {
        DepthAnything3::<WgpuBackend>::new(&device, config)
            .load_file(checkpoint, &recorder, &device)
    })
    .expect("failed to load checkpoint");
    println!("load finished.");
}
