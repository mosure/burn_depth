#![recursion_limit = "256"]

use std::path::Path;

use burn::{backend::Wgpu, nn::interpolate::InterpolateMode, prelude::*};
use burn_depth_pro::model::depth_pro::{DepthPro, DepthProConfig};
use half::f16;
use image::imageops::FilterType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = <Wgpu<f16> as Backend>::Device::default();
    let model = DepthPro::<Wgpu<f16>>::new(&device, DepthProConfig::default());

    let image_path = Path::new("assets/image/test.jpg");
    let image = image::open(image_path)
        .map_err(|err| format!("Failed to load image `{}`: {err}", image_path.display()))?;

    let image_size = model.img_size() as u32;
    let image = image
        .resize_exact(image_size, image_size, FilterType::Triangle)
        .to_rgb8();

    let mut data = Vec::with_capacity((3 * image_size * image_size) as usize);
    for channel in 0..3 {
        for y in 0..image_size {
            for x in 0..image_size {
                let pixel = image.get_pixel(x, y)[channel];
                data.push(pixel as f32 / 255.0);
            }
        }
    }

    let input: Tensor<Wgpu<f16>, 4> = Tensor::<Wgpu<f16>, 1>::from_floats(data.as_slice(), &device)
        .reshape([1, 3, image_size as usize, image_size as usize]);

    let result = model.infer(input, None, InterpolateMode::Linear);

    println!("depth shape: {:?}", result.depth.shape());
    println!("focal length (px): {:?}", result.focallength_px.to_data());

    Ok(())
}
