use dfdx::prelude::*;

/// ClippedReLU module
#[derive(Default, Debug, Clone, Copy)]
pub struct ClippedReLU;

impl ZeroSizedModule for ClippedReLU {}
impl NonMutableModule for ClippedReLU {}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for ClippedReLU {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        Ok(input.clamp(E::from_f32(0f32).unwrap(), E::from_f32(1f32).unwrap()))
    }
}
