use burn::{
    prelude::*,
    backend::Wgpu,
};
use criterion::{
    BenchmarkId,
    criterion_group,
    criterion_main,
    Criterion,
    Throughput,
};

use burn_depth_pro::model::depth_pro::DepthProConfig;


criterion_group!{
    name = depth_pro_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = inference_benchmark,
}
criterion_main!(depth_pro_benchmarks);


fn inference_benchmark(c: &mut Criterion) {
    let config = DepthProConfig::default_config();

    let mut group = c.benchmark_group("burn_depth_pro_inference");
    group.throughput(Throughput::Elements(1));
    group.bench_with_input(
        BenchmarkId::new("depth_pro", "default"),
        &config,
        |b, config| {
            let device = Default::default();
            let model = config.init(&device);
            let input: Tensor<Wgpu, 4> = Tensor::zeros(
                [1, config.input_channels, config.image_size, config.image_size],
                &device,
            );

            b.iter(|| model.forward(input.clone()).to_data());
        },
    );
}
