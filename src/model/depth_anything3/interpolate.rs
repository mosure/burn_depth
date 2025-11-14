use burn::{prelude::*, tensor::Tensor};

pub fn resize_bilinear<B: Backend>(
    input: Tensor<B, 4>,
    output_size: [usize; 2],
    align_corners: bool,
) -> Tensor<B, 4> {
    let [batch, channels, in_height, in_width] = input.shape().dims::<4>();
    let [out_height, out_width] = output_size;

    if in_height == out_height && in_width == out_width {
        return input;
    }

    let device = input.device();
    let data = input.into_data().convert::<f32>();
    let input_values = data
        .to_vec::<f32>()
        .expect("failed to read tensor values for interpolation");

    let mut output = vec![0.0f32; batch * channels * out_height * out_width];

    let scale_h = compute_scale(in_height, out_height, align_corners);
    let scale_w = compute_scale(in_width, out_width, align_corners);

    for b in 0..batch {
        for c in 0..channels {
            for y in 0..out_height {
                for x in 0..out_width {
                    let input_y = compute_source(y, scale_h, align_corners);
                    let input_x = compute_source(x, scale_w, align_corners);
                    let value = bilinear_sample(
                        &input_values,
                        in_width,
                        in_height,
                        input_x,
                        input_y,
                        b,
                        c,
                        channels,
                    );
                    let offset = ((((b * channels) + c) * out_height) + y) * out_width + x;
                    output[offset] = value;
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

fn compute_scale(input: usize, output: usize, align_corners: bool) -> f32 {
    if output <= 1 {
        0.0
    } else if align_corners {
        if input <= 1 {
            0.0
        } else {
            (input - 1) as f32 / (output - 1) as f32
        }
    } else {
        input as f32 / output as f32
    }
}

fn compute_source(index: usize, scale: f32, align_corners: bool) -> f32 {
    if align_corners {
        index as f32 * scale
    } else {
        (index as f32 + 0.5) * scale - 0.5
    }
}

fn bilinear_sample(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    x: f32,
    y: f32,
    batch: usize,
    channel: usize,
    channels: usize,
) -> f32 {
    let x0 = x.floor().max(0.0);
    let y0 = y.floor().max(0.0);
    let x1 = (x0 + 1.0).min((in_width - 1) as f32);
    let y1 = (y0 + 1.0).min((in_height - 1) as f32);

    let x0_idx = x0 as usize;
    let y0_idx = y0 as usize;
    let x1_idx = x1 as usize;
    let y1_idx = y1 as usize;

    let dx = x - x0;
    let dy = y - y0;

    let stride_c = in_height * in_width;
    let stride_b = channels * stride_c;

    let base = batch * stride_b + channel * stride_c;
    let top_left = input[base + y0_idx * in_width + x0_idx];
    let top_right = input[base + y0_idx * in_width + x1_idx];
    let bottom_left = input[base + y1_idx * in_width + x0_idx];
    let bottom_right = input[base + y1_idx * in_width + x1_idx];

    let top = top_left * (1.0 - dx) + top_right * dx;
    let bottom = bottom_left * (1.0 - dx) + bottom_right * dx;

    top * (1.0 - dy) + bottom * dy
}
