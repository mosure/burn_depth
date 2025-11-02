use burn::{
    prelude::*,
    backend::wgpu::Wgpu,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use image::{
    load_from_memory_with_format, DynamicImage, GenericImageView, ImageFormat,
};

use burn_depth_pro::model::depth_pro::{
    DepthPro,
    DepthProConfig,
};


// Placeholder for embedded model weights
// In production, this would be: include_bytes!("../assets/models/depth_pro.mpk")
static STATE_ENCODED: &[u8] = &[];

// Placeholder for test image
// In production, this would be: include_bytes!("../assets/images/test_0.png")
static INPUT_IMAGE_0: &[u8] = &[];


pub fn load_model<B: Backend>(
    config: &DepthProConfig,
    device: &B::Device,
) -> DepthPro<B> {
    if STATE_ENCODED.is_empty() {
        println!("Warning: No model weights embedded. Using uninitialized model.");
        return config.init(device);
    }

    let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .load(STATE_ENCODED.to_vec(), &Default::default())
        .expect("failed to decode state");

    let model = config.init(device);
    model.load_record(record)
}


fn center_crop(image: &DynamicImage, crop_width: u32, crop_height: u32) -> DynamicImage {
    let (img_width, img_height) = image.dimensions();

    let crop_width = crop_width.min(img_width);
    let crop_height = crop_height.min(img_height);

    let x = (img_width - crop_width) / 2;
    let y = (img_height - crop_height) / 2;

    image.crop_imm(x, y, crop_width, crop_height)
}

fn normalize<B: Backend>(
    input: Tensor<B, 4>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mean: Tensor<B, 1> = Tensor::from_floats([0.485, 0.456, 0.406], device);
    let std: Tensor<B, 1> = Tensor::from_floats([0.229, 0.224, 0.225], device);

    input
        .permute([0, 2, 3, 1])
        .sub(mean.unsqueeze())
        .div(std.unsqueeze())
        .permute([0, 3, 1, 2])
}

fn load_image<B: Backend>(
    bytes: &[u8],
    config: &DepthProConfig,
    device: &B::Device,
) -> Tensor<B, 4> {
    let img = load_from_memory_with_format(bytes, ImageFormat::Png)
        .unwrap()
        .resize_exact(
            config.image_size as u32 + 2,
            config.image_size as u32 + 2,
            image::imageops::FilterType::Lanczos3,
        );
    let img = center_crop(&img, config.image_size as u32, config.image_size as u32);

    let img_data: Vec<f32> = img.to_rgb32f()
        .pixels()
        .flat_map(|p| p.0)
        .collect();

    let input: Tensor<B, 1> = Tensor::from_floats(
        img_data.as_slice(),
        device,
    );

    let input = input.reshape([
        1,
        config.image_size,
        config.image_size,
        config.input_channels,
    ]);
    let input = input.permute([0, 3, 1, 2]);

    normalize(input, device)
}


fn main() {
    type Backend = Wgpu<f32, i32>;

    let device = Default::default();
    let config = DepthProConfig::default_config();

    println!("Loading model...");
    let model = load_model::<Backend>(&config, &device);

    if INPUT_IMAGE_0.is_empty() {
        println!("No test image available. Creating dummy input...");
        let input = Tensor::<Backend, 4>::zeros(
            [1, config.input_channels, config.image_size, config.image_size],
            &device,
        );
        
        println!("Running inference...");
        let output = model.forward(input);
        
        println!("Output shape: {:?}", output.dims());
        println!("Correctness check complete (with dummy data).");
    } else {
        println!("Loading test image...");
        let input = load_image::<Backend>(INPUT_IMAGE_0, &config, &device);

        println!("Running inference...");
        let output = model.forward(input);

        println!("Output shape: {:?}", output.dims());
        
        // TODO: Compare with expected output for correctness validation
        println!("Correctness check complete.");
    }
}
