# burn_depth 🔥📐😎

[![test](https://github.com/mosure/burn_depth/workflows/test/badge.svg)](https://github.com/Mosure/burn_depth/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/burn_depth.svg)](https://crates.io/crates/burn_depth)


burn [depth pro](https://github.com/apple/ml-depth-pro) model inference


| input               | metric depth               |
|-----------------------|-----------------------|
| ![Alt text](./assets/image/test.jpg)    | ![Alt text](./docs/test_depth.png)    |


## usage

```rust
use burn_depth::{
    DepthCheckpointSource, DepthLoadConfig, DepthModelKind, DepthPipeline,
    DepthPrecision, DepthRuntimeConfig, InferenceBackend,
};

let device = burn::tensor::Device::<InferenceBackend>::default();
let pipeline = DepthPipeline::<InferenceBackend>::load(
    &device,
    DepthLoadConfig {
        model: DepthModelKind::DepthPro,
        precision: DepthPrecision::F32,
        checkpoint: DepthCheckpointSource::Local("models/depth-pro/depth_pro.bpk".into()),
        cache_dir: None,
        allow_download: false,
        require_gpu: true,
    },
)?;

let image = image::open("assets/image/test.jpg")?;
let prediction = pipeline.predict(image, DepthRuntimeConfig::default())?;
// prediction.depth_m is metric depth. Depth Pro also fills focallength_px.
```

### switching between depth_pro and depth anything 3

```bash
cargo run --example inference -- \
  --model depth-pro \
  --checkpoint models/depth-pro/depth_pro.bpk \
  --image assets/image/test.jpg

cargo run --example inference -- \
  --model depth-anything3-metric-large \
  --checkpoint models/da3/da3_metric_large_f16.bpk \
  --precision f16 \
  --image assets/image/test.jpg
```

### sharded/CDN-ready artifacts

`DepthCheckpointSource::Local` loads a single `.bpk`. `DepthCheckpointSource::PartsManifest`
loads a local `.bpk.parts.json`, verifies every part SHA256, assembles atomically into the cache
directory, and verifies the full artifact hash. `DepthCheckpointSource::Cdn` is represented in
the public API, but fetching is intentionally caller-provided so native and wasm apps can use
their own async HTTP stack and progress events.

## setup

- download [`depth_pro.pt`](https://github.com/apple/ml-depth-pro/blob/main/get_pretrained_models.sh) to `assets/model/`
- `cargo run --bin import_depth_pro --features import -- --checkpoint assets/model/depth_pro.pt --output models/depth-pro/depth_pro.bpk --precision f32 --shard-size-mb 64`

- download [`da3_metric_large.safetensors`](https://huggingface.co/depth-anything/Depth-Anything-V3) to `assets/model/`
- `cargo run --bin import_da3 --features import -- --checkpoint assets/model/da3_metric_large.safetensors --output models/da3/da3_metric_large_f16.bpk --precision f16 --shard-size-mb 64`

- `cargo run --example inference -- --help`

## validation

```bash
cargo check --all-targets --no-default-features --features backend_wgpu
cargo test --no-default-features --features backend_ndarray
cargo test --no-default-features --features backend_wgpu -- --test-threads=1
cargo check --target wasm32-unknown-unknown --no-default-features --features backend_wgpu,wasm
BURN_DEPTH_LOADER_PARITY=1 cargo test --features backend_wgpu --test loader_parity -- --nocapture
```
