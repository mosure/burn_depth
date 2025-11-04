use burn::tensor::{Tensor, backend::Backend};

fn compute_output_size(input: usize, scale: f32) -> usize {
    let scaled = (input as f32 * scale).floor() as isize;
    scaled.max(1) as usize
}

fn bilinear_sample(input: &[f32], in_width: usize, in_height: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min((in_width - 1) as f32);
    let y1 = (y0 + 1.0).min((in_height - 1) as f32);

    let x0_idx = x0.max(0.0) as usize;
    let y0_idx = y0.max(0.0) as usize;
    let x1_idx = x1 as usize;
    let y1_idx = y1 as usize;

    let dx = x - x0;
    let dy = y - y0;

    let top_left = input[y0_idx * in_width + x0_idx];
    let top_right = input[y0_idx * in_width + x1_idx];
    let bottom_left = input[y1_idx * in_width + x0_idx];
    let bottom_right = input[y1_idx * in_width + x1_idx];

    let top = top_left * (1.0 - dx) + top_right * dx;
    let bottom = bottom_left * (1.0 - dx) + bottom_right * dx;

    top * (1.0 - dy) + bottom * dy
}

pub fn resize_bilinear_align_corners_false<B: Backend>(
    input: Tensor<B, 4>,
    output_size: [usize; 2],
) -> Tensor<B, 4> {
    let [batch, channels, in_height, in_width] = input.shape().dims::<4>();
    let [out_height, out_width] = output_size;

    if in_height == out_height && in_width == out_width {
        return input;
    }

    assert!(
        out_height > 0 && out_width > 0,
        "output size must be positive"
    );

    let device = input.device();
    let data = input.into_data().convert::<f32>();
    let input_values = data
        .to_vec::<f32>()
        .expect("failed to convert tensor data to Vec<f32>");

    let mut output = vec![0.0f32; batch * channels * out_height * out_width];

    let scale_y = in_height as f32 / out_height as f32;
    let scale_x = in_width as f32 / out_width as f32;

    for b in 0..batch {
        for c in 0..channels {
            let input_offset = (b * channels + c) * in_height * in_width;
            let output_offset = (b * channels + c) * out_height * out_width;

            for oy in 0..out_height {
                let in_y = (oy as f32 + 0.5) * scale_y - 0.5;
                for ox in 0..out_width {
                    let in_x = (ox as f32 + 0.5) * scale_x - 0.5;

                    let value = bilinear_sample(
                        &input_values[input_offset..input_offset + in_height * in_width],
                        in_width,
                        in_height,
                        in_x,
                        in_y,
                    );
                    output[output_offset + oy * out_width + ox] = value;
                }
            }
        }
    }

    Tensor::<B, 1>::from_floats(output.as_slice(), &device).reshape([
        batch as i32,
        channels as i32,
        out_height as i32,
        out_width as i32,
    ])
}

pub fn resize_bilinear_scale<B: Backend>(input: Tensor<B, 4>, scale: [f32; 2]) -> Tensor<B, 4> {
    let dims = input.shape().dims::<4>();
    let target_height = compute_output_size(dims[2], scale[0]);
    let target_width = compute_output_size(dims[3], scale[1]);
    resize_bilinear_align_corners_false(input, [target_height, target_width])
}
