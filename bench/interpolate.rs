#![recursion_limit = "256"]

use burn::prelude::*;
use burn_depth::{
    InferenceBackend,
    model::depth_pro::{InterpolationMethod, resize_bilinear_align_corners_false},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

criterion_group! {
    name = interpolation_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = interpolation_benchmark,
}
criterion_main!(interpolation_benchmarks);

fn interpolation_benchmark(c: &mut Criterion) {
    let device = <InferenceBackend as Backend>::Device::default();
    let bench_device = device.clone();

    struct ResizeCase {
        name: &'static str,
        channels: usize,
        batch: usize,
        in_height: usize,
        in_width: usize,
        out_height: usize,
        out_width: usize,
    }

    let cases = [
        ResizeCase {
            name: "c3_b1_360x540_to_1536x1536",
            channels: 3,
            batch: 1,
            in_height: 360,
            in_width: 540,
            out_height: 1536,
            out_width: 1536,
        },
        ResizeCase {
            name: "c3_b1_1536x1536_to_768x768",
            channels: 3,
            batch: 1,
            in_height: 1536,
            in_width: 1536,
            out_height: 768,
            out_width: 768,
        },
        ResizeCase {
            name: "c3_b1_1536x1536_to_384x384_case1",
            channels: 3,
            batch: 1,
            in_height: 1536,
            in_width: 1536,
            out_height: 384,
            out_width: 384,
        },
        ResizeCase {
            name: "c3_b1_1536x1536_to_384x384_case2",
            channels: 3,
            batch: 1,
            in_height: 1536,
            in_width: 1536,
            out_height: 384,
            out_width: 384,
        },
        ResizeCase {
            name: "c1_b1_1536x1536_to_360x540",
            channels: 1,
            batch: 1,
            in_height: 1536,
            in_width: 1536,
            out_height: 360,
            out_width: 540,
        },
    ];

    let mut group = c.benchmark_group("burn_depth_interpolation");
    for case in cases {
        let input: Tensor<InferenceBackend, 4> = Tensor::zeros(
            [case.batch, case.channels, case.in_height, case.in_width],
            &device,
        );
        let throughput = (case.batch * case.channels * case.out_height * case.out_width) as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_function(format!("{}::custom", case.name), |b| {
            b.iter(|| {
                let output = resize_bilinear_align_corners_false(
                    input.clone(),
                    [case.out_height, case.out_width],
                    InterpolationMethod::Custom,
                );
                InferenceBackend::sync(&bench_device);
                black_box(output);
            });
        });

        group.bench_function(format!("{}::burn", case.name), |b| {
            b.iter(|| {
                let output = resize_bilinear_align_corners_false(
                    input.clone(),
                    [case.out_height, case.out_width],
                    InterpolationMethod::Burn,
                );
                InferenceBackend::sync(&bench_device);
                black_box(output);
            });
        });
    }
    group.finish();
}
