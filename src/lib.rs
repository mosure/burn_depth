#![recursion_limit = "256"]
#![allow(deprecated)]

pub mod geometry;
pub mod inference;
pub mod loader;
pub mod model;
pub mod pipeline;

pub use geometry::{
    CameraIntrinsics, ImageBoundingBox, Plane, backproject_depth, depth_at_bbox_contact_region,
    estimate_floor_plane, pixel_to_ray,
};
pub use loader::{
    DepthArtifactManifest, DepthArtifactPart, DepthCheckpointSource, DepthLoadConfig,
    DepthLoadError, DepthLoadEvent, DepthLoadStage, DepthPrecision,
};
pub use model::DepthModelKind;
pub use pipeline::{DepthPipeline, DepthPipelineError, DepthRuntimeConfig};

#[cfg(feature = "backend_wgpu")]
pub type InferenceBackend = burn::backend::Wgpu<f32>;

#[cfg(all(not(feature = "backend_wgpu"), feature = "backend_cuda"))]
pub type InferenceBackend = burn::backend::Cuda;

#[cfg(all(
    not(any(feature = "backend_wgpu", feature = "backend_cuda")),
    feature = "backend_ndarray"
))]
pub type InferenceBackend = burn::backend::NdArray;

#[cfg(all(
    not(any(
        feature = "backend_wgpu",
        feature = "backend_cuda",
        feature = "backend_ndarray"
    )),
    feature = "backend_cpu"
))]
pub type InferenceBackend = burn::backend::Cpu;

#[cfg(test)]
mod tests {
    use super::model::depth_pro::{DepthPro, DepthProConfig, layers::vit::DINOV2_L16_128};

    #[cfg(feature = "backend_cuda")]
    use burn::backend::Cuda as CudaBackend;

    #[cfg(feature = "backend_ndarray")]
    use burn::backend::NdArray as NdArrayBackend;

    #[cfg(feature = "backend_wgpu")]
    use burn::backend::wgpu::{RuntimeOptions, graphics::AutoGraphicsApi, init_setup};

    use burn::prelude::*;
    use std::any::type_name;
    use std::panic::{self, AssertUnwindSafe};

    #[cfg(feature = "backend_wgpu")]
    use std::sync::OnceLock;
    #[cfg(feature = "backend_wgpu")]
    use wgpu::Features;

    #[cfg(feature = "backend_wgpu")]
    use half::f16;

    #[cfg(feature = "backend_wgpu")]
    type WgpuHalfBackend = burn::backend::Wgpu<f16>;
    #[cfg(feature = "backend_wgpu")]
    type WgpuF32Backend = burn::backend::Wgpu<f32>;

    #[cfg(feature = "backend_wgpu")]
    static WGPU_FEATURES: OnceLock<Result<Features, String>> = OnceLock::new();

    #[cfg(feature = "backend_wgpu")]
    fn ensure_wgpu_runtime() -> Result<Features, String> {
        WGPU_FEATURES
            .get_or_init(|| {
                let device = burn::tensor::Device::<WgpuF32Backend>::default();
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
    fn init_wgpu_f16_device() -> Result<burn::tensor::Device<WgpuHalfBackend>, String> {
        let features = ensure_wgpu_runtime()?;

        if !features.contains(Features::SHADER_F16) {
            return Err("adapter does not expose SHADER_F16".to_string());
        }

        Ok(burn::tensor::Device::<WgpuHalfBackend>::default())
    }

    #[cfg(feature = "backend_wgpu")]
    fn init_wgpu_f32_device() -> Result<burn::tensor::Device<WgpuF32Backend>, String> {
        ensure_wgpu_runtime()?;
        Ok(burn::tensor::Device::<WgpuF32Backend>::default())
    }

    #[cfg(feature = "backend_cuda")]
    fn init_cuda_device() -> Result<burn::tensor::Device<CudaBackend<f32>>, String> {
        panic::catch_unwind(AssertUnwindSafe(|| {
            burn::tensor::Device::<CudaBackend<f32>>::default()
        }))
        .map_err(|_| "CUDA runtime unavailable on this system.".to_string())
    }

    #[cfg(feature = "backend_ndarray")]
    fn init_ndarray_device() -> Result<burn::tensor::Device<NdArrayBackend<f32>>, String> {
        Ok(burn::tensor::Device::<NdArrayBackend<f32>>::default())
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
        let result = model.infer(input);

        assert_eq!(result.depth.shape().dims(), [1, size, size]);
        assert_eq!(result.focallength_px.shape().dims(), [1]);
    }

    #[cfg(feature = "backend_ndarray")]
    fn heavy_inference_enabled() -> bool {
        std::env::var("BURN_DEPTH_HEAVY_INFERENCE")
            .map(|value| value != "0")
            .unwrap_or(false)
    }

    #[test]
    #[cfg(feature = "backend_wgpu")]
    fn depth_pro_initializes_wgpu_f16() {
        run_initializes_test::<WgpuHalfBackend, _>(
            init_wgpu_f16_device,
            Availability::Optional("WGPU<f16> backend test"),
        );
    }

    #[test]
    #[cfg(feature = "backend_wgpu")]
    fn depth_pro_roundtrip_record_wgpu_f16() {
        run_roundtrip_test::<WgpuHalfBackend, _>(
            init_wgpu_f16_device,
            Availability::Optional("WGPU<f16> backend test"),
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
            Availability::Optional("WGPU<f16> backend test"),
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
        if !heavy_inference_enabled() {
            eprintln!(
                "skipping full DepthPro ndarray inference; set BURN_DEPTH_HEAVY_INFERENCE=1 to run"
            );
            return;
        }

        run_inference_test::<NdArrayBackend<f32>, _>(
            init_ndarray_device,
            Availability::Required("NdArray backend unavailable"),
        );
    }
}
