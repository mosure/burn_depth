use image::{Rgb, RgbImage};

fn cubic_weight(x: f32) -> f32 {
    const A: f32 = -0.75;
    let abs = x.abs();
    let abs2 = abs * abs;
    let abs3 = abs2 * abs;
    if abs <= 1.0 {
        (A + 2.0) * abs3 - (A + 3.0) * abs2 + 1.0
    } else if abs < 2.0 {
        A * abs3 - 5.0 * A * abs2 + 8.0 * A * abs - 4.0 * A
    } else {
        0.0
    }
}

fn clamp_index(value: isize, max: isize) -> usize {
    value.clamp(0, max) as usize
}

pub fn resize_with_bicubic(
    image: &RgbImage,
    target_width: u32,
    target_height: u32,
) -> Result<RgbImage, String> {
    if target_width == 0 || target_height == 0 {
        return Err("Target dimensions must be positive.".to_string());
    }
    if image.width() == target_width && image.height() == target_height {
        return Ok(image.clone());
    }

    let src_width = image.width() as usize;
    let src_height = image.height() as usize;
    let dst_width = target_width as usize;
    let dst_height = target_height as usize;
    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;
    let mut output = RgbImage::new(target_width, target_height);

    for dst_y in 0..dst_height {
        let src_y = (dst_y as f32 + 0.5) * scale_y - 0.5;
        let y_int = src_y.floor() as isize;
        for dst_x in 0..dst_width {
            let src_x = (dst_x as f32 + 0.5) * scale_x - 0.5;
            let x_int = src_x.floor() as isize;
            let mut accum = [0.0f32; 3];
            let mut weight_sum = 0.0f32;

            for m in -1..=2 {
                let wy = cubic_weight(src_y - (y_int + m) as f32);
                let sy = clamp_index(y_int + m, (src_height - 1) as isize);
                for n in -1..=2 {
                    let wx = cubic_weight(src_x - (x_int + n) as f32);
                    let sx = clamp_index(x_int + n, (src_width - 1) as isize);
                    let weight = wy * wx;
                    weight_sum += weight;
                    let pixel = image.get_pixel(sx as u32, sy as u32);
                    for channel in 0..3 {
                        accum[channel] += weight * pixel[channel] as f32;
                    }
                }
            }

            if weight_sum.abs() > 1e-6 {
                for channel in 0..3 {
                    accum[channel] /= weight_sum;
                }
            }
            let mut rgb = [0u8; 3];
            for channel in 0..3 {
                rgb[channel] = accum[channel].round().clamp(0.0, 255.0) as u8;
            }
            output.put_pixel(dst_x as u32, dst_y as u32, Rgb(rgb));
        }
    }

    Ok(output)
}
