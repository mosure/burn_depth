#![recursion_limit = "256"]

pub mod model;

#[cfg(test)]
mod tests {
    use super::model::depth_pro::{DepthPro, DepthProConfig};
    use burn::prelude::*;
    use burn_wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions};
    use half::f16;
    use wgpu::Features;
    use std::panic;

    type TestBackend = burn_wgpu::Wgpu<f16>;

    fn init_device_with_f16() -> <TestBackend as Backend>::Device {
        let device = <TestBackend as Backend>::Device::default();

        let setup = panic::catch_unwind(|| init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default()))
            .unwrap_or_else(|_| panic!("Failed to initialize WGPU runtime for f16 tests."));

        if !setup.adapter.features().contains(Features::SHADER_F16) {
            panic!(
                "Adapter {:?} does not expose SHADER_F16; f16 backend tests require this feature.",
                setup.adapter.get_info()
            );
        }

        device
    }

    fn build_model(device: &<TestBackend as Backend>::Device) -> DepthPro<TestBackend> {
        panic::catch_unwind(|| DepthPro::<TestBackend>::new(device, DepthProConfig::default()))
            .unwrap_or_else(|_| {
                panic!("DepthPro initialization panicked when using WGPU<f16> backend.");
            })
    }

    #[test]
    fn depth_pro_initializes() {
        let device = init_device_with_f16();
        let model = build_model(&device);
        assert!(model.img_size() > 0);
        drop(model);
    }

    #[test]
    fn depth_pro_roundtrip_record() {
        let device = init_device_with_f16();
        let model = build_model(&device);
        let record = model.clone().into_record();

        let reloaded = build_model(&device).load_record(record);

        assert_eq!(model.img_size(), reloaded.img_size());
    }
}
