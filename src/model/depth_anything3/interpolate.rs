use burn::{
    nn::interpolate::{Interpolate2dConfig, InterpolateMode},
    prelude::*,
    tensor::Tensor,
};

pub fn resize_bilinear<B: Backend>(
    input: Tensor<B, 4>,
    output_size: [usize; 2],
    _align_corners: bool,
) -> Tensor<B, 4> {
    let dims = input.shape().dims::<4>();
    let batch = dims[0];
    let channels = dims[1];
    let in_height = dims[2];
    let in_width = dims[3];
    let device = input.device();

    let contiguous = Tensor::<B, 4>::zeros(
        [
            batch as i32,
            channels as i32,
            in_height as i32,
            in_width as i32,
        ],
        &device,
    )
    .slice_assign(
        [
            0..batch as i32,
            0..channels as i32,
            0..in_height as i32,
            0..in_width as i32,
        ],
        input.clone(),
    );

    if [in_height, in_width] == output_size {
        return contiguous;
    }

    Interpolate2dConfig::new()
        .with_output_size(Some(output_size))
        .with_mode(InterpolateMode::Linear)
        .init()
        .forward(contiguous)
}
