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
    DepthLoadConfig, DepthModelKind, DepthPipeline, DepthPrecision,
    DepthRuntimeConfig, InferenceBackend,
};

let device = burn::tensor::Device::<InferenceBackend>::default();
let pipeline = DepthPipeline::<InferenceBackend>::load(
    &device,
    DepthLoadConfig::cdn(DepthModelKind::DepthPro, DepthPrecision::F32),
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
directory, and verifies the full artifact hash.

`DepthLoadConfig::cdn(model, precision)` uses `DepthCheckpointSource::Cdn` with:

- default base URL: `https://aberration.technology/model/burn_depth`
- default cache directory: `$HOME/.burn_depth`
- default manifests:
  - `depth-pro/depth_pro.bpk.parts.json`
  - `depth-pro/depth_pro_f16.bpk.parts.json`
  - `da3/da3_metric_large.bpk.parts.json`
  - `da3/da3_metric_large_f16.bpk.parts.json`

For example, DA3 metric large f16 resolves to:

```text
https://aberration.technology/model/burn_depth/da3/da3_metric_large_f16.bpk.parts.json
```

Each part URL is resolved relative to the manifest URL, so a manifest entry named
`da3_metric_large_f16.part-00000.bpk` is fetched from:

```text
https://aberration.technology/model/burn_depth/da3/da3_metric_large_f16.part-00000.bpk
```

Native `DepthPipeline::load` downloads the manifest and missing/stale parts into
`cache_dir.unwrap_or_else(default_cache_dir)` using partial files, verifies SHA256/byte lengths,
assembles the full `.bpk`, and reuses the cached model when `allow_download` is false. The default
native cache directory is `$HOME/.burn_depth`.

Browser wasm callers should use `DepthPipeline::load_async` or
`resolve_checkpoint_bytes_async`. The wasm loader uses `fetch` plus browser `CacheStorage` cache
`burn_depth:model-shards:v1`, caches the manifest and each part by URL, verifies cached bytes before
use, refetches corrupt entries when `allow_download` is true, assembles the `.bpk` bytes in memory,
and deserializes the model from BurnPack bytes. It does not use synchronous XHR or synchronous file
paths. Native `load_async` delegates back to the file-backed loader so native performance stays on
the same path as `load`.

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
