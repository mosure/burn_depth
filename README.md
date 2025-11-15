# burn_depth 🔥📐😎

[![test](https://github.com/mosure/burn_depth/workflows/test/badge.svg)](https://github.com/Mosure/burn_depth/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/burn_depth.svg)](https://crates.io/crates/burn_depth)


burn [depth pro](https://github.com/apple/ml-depth-pro) model inference


| input               | metric depth               |
|-----------------------|-----------------------|
| ![Alt text](./assets/image/test.jpg)    | ![Alt text](./docs/test_depth.png)    |


## usage

```rust
use burn::prelude::*;
use burn_depth::{InferenceBackend, model::depth_pro::DepthPro};

let device = <InferenceBackend as Backend>::Device::default();

let model = DepthPro::<InferenceBackend>::load(&device, "assets/model/depth_pro.mpk")?;

// Image tensor with shape [1, 3, H, W] (batch, channels, height, width)
let input: Tensor<InferenceBackend, 4> = Tensor::zeros([1, 3, 512, 512], &device);

let result = model.infer(input, None);
// result.depth: Tensor<InferenceBackend, 3> with shape [1, H, W]
// result.focallength_px: Tensor<InferenceBackend, 1> with shape [1]
```

### switching between depth_pro and depth anything 3

```bash
cargo run --example inference -- \
  --model depth-pro \
  --checkpoint assets/model/depth_pro.mpk \
  --image assets/image/test.jpg

cargo run --example inference -- \
  --model depth-anything3 \
  --checkpoint assets/model/da3_metric_large.mpk \
  --image assets/image/test.jpg
```


## setup

- download [`depth_pro.pt`](https://github.com/apple/ml-depth-pro/blob/main/get_pretrained_models.sh) to `assets/model/`
- `cargo run --bin import_depth_pro --features import`

- download [`da3_metric_large.safetensors`](https://huggingface.co/depth-anything/Depth-Anything-V3) to `assets/model/`
- `cargo run --bin import_da3 --features import`

- `cargo run --example inference -- --help`
