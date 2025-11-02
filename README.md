# burn_depth_pro 🔥📐

[![GitHub License](https://img.shields.io/github/license/mosure/burn_depth_pro)](https://raw.githubusercontent.com/mosure/burn_depth_pro/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/burn_depth_pro)](https://github.com/mosure/burn_depth_pro)
[![crates.io](https://img.shields.io/crates/v/burn_depth_pro.svg)](https://crates.io/crates/burn_depth_pro)

burn [Depth Pro](https://github.com/apple/ml-depth-pro) model inference 🔥📐😎

Monocular depth estimation using Apple's Depth Pro model implemented in the Burn deep learning framework.


## Features

- [ ] Inference
- [ ] PyTorch model import tooling
- [ ] Trace comparison with PyTorch
- [ ] Benchmarking
- [ ] Optimized encoder/decoder
- [ ] Automatic weights cache/download
- [ ] Quantization


## Setup

### Prerequisites
- Rust (latest stable)
- Python 3.8+ (for model import and comparison)

### Download Pre-trained Model
1. Download the Depth Pro model from [Apple's ML Depth Pro repository](https://github.com/apple/ml-depth-pro)
2. Place the model weights in `./assets/models/`
3. Run the import tool:
   ```bash
   cargo run --bin import
   ```

### Running Examples

#### Correctness Check
Compare Burn implementation with PyTorch reference:
```bash
# First, generate reference outputs with PyTorch
python tool/standard.py

# Then run the Burn implementation
cargo run --example correctness
```

#### Benchmarking
```bash
cargo bench
# Open target/criterion/report/index.html to view results
```


## Project Structure

```
burn_depth_pro/
├── src/
│   ├── lib.rs              # Library root
│   ├── model/              # Model implementations
│   │   └── depth_pro.rs    # Depth Pro model
│   └── layers/             # Neural network layers
│       └── mod.rs          # Layer modules
├── tool/
│   ├── import.rs           # PyTorch → Burn model import
│   ├── benchmark.rs        # Performance benchmarking
│   └── standard.py         # PyTorch reference implementation
├── example/
│   └── correctness.rs      # Correctness validation
└── assets/
    ├── models/             # Model weights (not committed)
    ├── images/             # Test images (not committed)
    └── tensors/            # Saved tensors (not committed)
```


## License

MIT OR Apache-2.0
