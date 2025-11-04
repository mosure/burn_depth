# burn_depth 🔥📐😎

[![test](https://github.com/mosure/burn_depth/workflows/test/badge.svg)](https://github.com/Mosure/burn_depth/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/burn_depth.svg)](https://crates.io/crates/burn_depth)


burn [depth pro](https://github.com/apple/ml-depth-pro) model inference


| input               | metric depth               |
|-----------------------|-----------------------|
| ![Alt text](./assets/image/test.jpg)    | ![Alt text](./docs/test_depth.png)    |


## usage

```rust
use burn_depth::model::depth_pro::DepthPro;

let model = DepthPro::<InferenceBackend>::load("assets/model/depth_pro.mpk")?;
let depth = model.forward(input);
```


## setup

- download [`depth_pro.pt`](https://github.com/apple/ml-depth-pro/blob/main/get_pretrained_models.sh) to `assets/model/`
- `cargo run --bin import --features import`
- `cargo run --example inference`
