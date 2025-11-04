#![recursion_limit = "256"]

use burn::prelude::*;
use burn_depth::{
    InferenceBackend,
    model::depth_pro::{DepthPro, DepthProConfig},
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
    let model = DepthPro::<InferenceBackend>::new(&device, DepthProConfig::default());
    let image_size = model.img_size();
    let input: Tensor<InferenceBackend, 4> = Tensor::zeros([1, 3, image_size, image_size], &device);
    let bench_device = device.clone();

    let mut group = c.benchmark_group("burn_depth_inference");
    group.throughput(Throughput::Elements(1));
    group.bench_function("depth_pro_infer", |b| {
        b.iter(|| {
            let output = model.infer(input.clone(), None);
            InferenceBackend::sync(&bench_device);
            black_box(output);
        });
    });
    group.finish();
}
