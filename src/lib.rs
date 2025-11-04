#![recursion_limit = "256"]

pub mod inference;
pub mod model;

#[cfg(test)]
mod tests {
    use super::model::depth_pro::{DepthPro, DepthProConfig, layers::vit::DINOV2_L16_128};

    #[cfg(feature = "backend_cuda")]
    use burn::backend::Cuda as CudaBackend;

    #[cfg(feature = "backend_ndarray")]
    use burn::backend::NdArray as NdArrayBackend;

    #[cfg(feature = "backend_wgpu")]
    use burn::backend::{
        Wgpu as WgpuBackend,
        wgpu::{RuntimeOptions, graphics::AutoGraphicsApi, init_setup},
    };

    use burn::{nn::interpolate::InterpolateMode, prelude::*};
    use std::any::type_name;
    use std::panic::{self, AssertUnwindSafe};

    #[cfg(feature = "backend_wgpu")]
    use std::sync::OnceLock;
    #[cfg(feature = "backend_wgpu")]
    use wgpu::Features;

    #[cfg(feature = "backend_wgpu")]
    use half::f16;

    #[cfg(feature = "backend_wgpu")]
    type WgpuHalfBackend = Wgpu<f16>;
    #[cfg(feature = "backend_wgpu")]
    type WgpuF32Backend = Wgpu<f32>;

    #[cfg(feature = "backend_wgpu")]
    static WGPU_FEATURES: OnceLock<Result<Features, String>> = OnceLock::new();

    #[cfg(feature = "backend_wgpu")]
    fn ensure_wgpu_runtime() -> Result<Features, String> {
        WGPU_FEATURES
            .get_or_init(|| {
                let device = <WgpuF32Backend as Backend>::Device::default();
                match panic::catch_unwind(AssertUnwindSafe(|| {
                    init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default())
                })) {
                    Ok(setup) => Ok(setup.adapter.features()),
                    Err(_) => Err("Failed to initialize WGPU runtime for tests.".to_string()),
                }
            })
            .clone()
    }

    #[cfg(feature = "backend_wgpu")]
    fn init_wgpu_f16_device() -> Result<<WgpuHalfBackend as Backend>::Device, String> {
        let features = ensure_wgpu_runtime()?;

        if !features.contains(Features::SHADER_F16) {
            return Err("adapter does not expose SHADER_F16".to_string());
        }

        Ok(<WgpuHalfBackend as Backend>::Device::default())
    }

    #[cfg(feature = "backend_wgpu")]
    fn init_wgpu_f32_device() -> Result<<WgpuF32Backend as Backend>::Device, String> {
        ensure_wgpu_runtime()?;
        Ok(<WgpuF32Backend as Backend>::Device::default())
    }

    #[cfg(feature = "backend_cuda")]
    fn init_cuda_device() -> Result<<CudaBackend<f32> as Backend>::Device, String> {
        panic::catch_unwind(AssertUnwindSafe(|| {
            <CudaBackend<f32> as Backend>::Device::default()
        }))
        .map_err(|_| "CUDA runtime unavailable on this system.".to_string())
    }

    #[cfg(feature = "backend_ndarray")]
    fn init_ndarray_device() -> Result<<NdArrayBackend<f32> as Backend>::Device, String> {
        Ok(<NdArrayBackend<f32> as Backend>::Device::default())
    }

    fn test_config() -> DepthProConfig {
        // Use the 128 window preset to reduce the full-resolution input (512 px) so the
        // backend sweep stays fast while still exercising the multi-scale pipeline.
        DepthProConfig {
            patch_encoder_preset: DINOV2_L16_128.into(),
            image_encoder_preset: DINOV2_L16_128.into(),
            fov_encoder_preset: Some(DINOV2_L16_128.into()),
            decoder_features: 64,
            ..DepthProConfig::default()
        }
    }

    fn build_model<B: Backend>(device: &B::Device) -> DepthPro<B> {
        panic::catch_unwind(AssertUnwindSafe(|| {
            DepthPro::<B>::new(device, test_config())
        }))
        .unwrap_or_else(|_| {
            panic!(
                "DepthPro initialization panicked when using backend `{}`.",
                type_name::<B>()
            );
        })
    }

    #[allow(dead_code)]
    #[derive(Clone, Copy)]
    enum Availability {
        Optional(&'static str),
        Required(&'static str),
    }

    fn resolve_device<B, F>(make_device: F, availability: Availability) -> Option<B::Device>
    where
        B: Backend,
        F: Fn() -> Result<B::Device, String>,
    {
        match make_device() {
            Ok(device) => Some(device),
            Err(reason) => match availability {
                Availability::Optional(label) => {
                    println!("ignored {label}: {reason}");
                    None
                }
                Availability::Required(label) => panic!("{label}: {reason}"),
            },
        }
    }

    fn run_initializes_test<B, F>(make_device: F, availability: Availability)
    where
        B: Backend,
        F: Fn() -> Result<B::Device, String>,
    {
        let Some(device) = resolve_device::<B, _>(make_device, availability) else {
            return;
        };

        let model = build_model::<B>(&device);
        assert!(model.img_size() > 0);
    }

    fn run_roundtrip_test<B, F>(make_device: F, availability: Availability)
    where
        B: Backend,
        F: Fn() -> Result<B::Device, String>,
    {
        let Some(device) = resolve_device::<B, _>(make_device, availability) else {
            return;
        };

        let model = build_model::<B>(&device);
        let record = model.clone().into_record();
        let reloaded = build_model::<B>(&device).load_record(record);

        assert_eq!(model.img_size(), reloaded.img_size());
    }

    fn run_inference_test<B, F>(make_device: F, availability: Availability)
    where
        B: Backend,
        F: Fn() -> Result<B::Device, String>,
    {
        let Some(device) = resolve_device::<B, _>(make_device, availability) else {
            return;
        };

        let model = build_model::<B>(&device);
        let size = model.img_size();
        let input = Tensor::<B, 4>::zeros([1, 3, size, size], &device);
        let result = model.infer(input, None, InterpolateMode::Nearest);

        assert_eq!(result.depth.shape().dims(), [1, size, size]);
        assert_eq!(result.focallength_px.shape().dims(), [1]);
    }

    #[test]
    #[cfg(feature = "backend_wgpu")]
    fn depth_pro_initializes_wgpu_f16() {
        run_initializes_test::<WgpuHalfBackend, _>(
            init_wgpu_f16_device,
            Availability::Required("WGPU<f16> backend unavailable"),
        );
    }

    #[test]
    #[cfg(feature = "backend_wgpu")]
    fn depth_pro_roundtrip_record_wgpu_f16() {
        run_roundtrip_test::<WgpuHalfBackend, _>(
            init_wgpu_f16_device,
            Availability::Required("WGPU<f16> backend unavailable"),
        );
    }

    #[test]
    #[cfg(feature = "backend_wgpu")]
    fn depth_pro_initializes_wgpu_f32() {
        run_initializes_test::<WgpuF32Backend, _>(
            init_wgpu_f32_device,
            Availability::Optional("WGPU<f32> backend test"),
        );
    }

    #[test]
    #[cfg(feature = "backend_wgpu")]
    fn depth_pro_roundtrip_record_wgpu_f32() {
        run_roundtrip_test::<WgpuF32Backend, _>(
            init_wgpu_f32_device,
            Availability::Optional("WGPU<f32> backend test"),
        );
    }

    #[test]
    #[cfg(feature = "backend_cuda")]

    fn depth_pro_initializes_cuda() {
        run_initializes_test::<CudaBackend<f32>, _>(
            init_cuda_device,
            Availability::Required("CUDA backend unavailable"),
        );
    }

    #[test]
    #[cfg(feature = "backend_cuda")]
    fn depth_pro_roundtrip_record_cuda() {
        run_roundtrip_test::<CudaBackend<f32>, _>(
            init_cuda_device,
            Availability::Required("CUDA backend unavailable"),
        );
    }

    #[test]
    #[cfg(feature = "backend_ndarray")]
    fn depth_pro_initializes_ndarray() {
        run_initializes_test::<NdArrayBackend<f32>, _>(
            init_ndarray_device,
            Availability::Required("NdArray backend unavailable"),
        );
    }

    #[test]
    #[cfg(feature = "backend_ndarray")]
    fn depth_pro_roundtrip_record_ndarray() {
        run_roundtrip_test::<NdArrayBackend<f32>, _>(
            init_ndarray_device,
            Availability::Required("NdArray backend unavailable"),
        );
    }

    #[test]
    #[cfg(feature = "backend_wgpu")]
    fn depth_pro_infers_wgpu_f16() {
        run_inference_test::<WgpuHalfBackend, _>(
            init_wgpu_f16_device,
            Availability::Required("WGPU<f16> backend unavailable"),
        );
    }

    #[test]
    #[cfg(feature = "backend_wgpu")]
    fn depth_pro_infers_wgpu_f32() {
        run_inference_test::<WgpuF32Backend, _>(
            init_wgpu_f32_device,
            Availability::Optional("WGPU<f32> backend test"),
        );
    }

    #[test]
    #[cfg(feature = "backend_cuda")]
    fn depth_pro_infers_cuda() {
        run_inference_test::<CudaBackend<f32>, _>(
            init_cuda_device,
            Availability::Required("CUDA backend unavailable"),
        );
    }

    #[test]
    #[cfg(feature = "backend_ndarray")]
    fn depth_pro_infers_ndarray() {
        run_inference_test::<NdArrayBackend<f32>, _>(
            init_ndarray_device,
            Availability::Required("NdArray backend unavailable"),
        );
    }
}
