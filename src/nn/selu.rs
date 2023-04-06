use dfdx::{
    data::*, losses::mse_loss, nn::SaveToNpz, optim::*, prelude::*, tensor::*, tensor_ops::Backward,
};

const SCALE: f32 = 1.0507009873554804934193349852946;
const ALPHA: f32 = 1.6732632423543772848170429916717;

/// SeLU module
#[derive(Default, Debug, Clone, Copy)]
struct SeLU;

impl<S, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for SeLU
where
    S: Shape,
{
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        //let x = input.
		todo!()
    }
}