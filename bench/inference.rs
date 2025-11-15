#![recursion_limit = "256"]

use burn::prelude::*;
use burn_depth::{
    InferenceBackend,
    model::{
        depth_anything3::{DepthAnything3, DepthAnything3Config},
        depth_pro::{DepthPro, DepthProConfig},
    },
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

criterion_group! {
    name = depth_pro_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = inference_benchmark,
}
criterion_main!(depth_pro_benchmarks);

fn inference_benchmark(c: &mut Criterion) {
    let device = <InferenceBackend as Backend>::Device::default();
    let bench_device = device.clone();

    let depth_pro = DepthPro::<InferenceBackend>::new(&device, DepthProConfig::default());
    let pro_size = depth_pro.img_size();
    let pro_input = Tensor::<InferenceBackend, 4>::zeros([1, 3, pro_size, pro_size], &device);

    let depth_anything =
        DepthAnything3::<InferenceBackend>::new(&device, DepthAnything3Config::metric_large());
    let da3_size = depth_anything.img_size();
    let da3_input = Tensor::<InferenceBackend, 4>::zeros([1, 3, da3_size, da3_size], &device);

    let mut group = c.benchmark_group("burn_depth_inference");
    group.throughput(Throughput::Elements(1));
    group.bench_function("depth_pro_infer", |b| {
        b.iter(|| {
            let output = depth_pro.infer(pro_input.clone());
            InferenceBackend::sync(&bench_device);
            black_box(output);
        });
    });
    group.bench_function("depth_anything3_infer", |b| {
        b.iter(|| {
            let output = depth_anything.infer(da3_input.clone());
            InferenceBackend::sync(&bench_device);
            black_box(output);
        });
    });
    group.finish();
}
