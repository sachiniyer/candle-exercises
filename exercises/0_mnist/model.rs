// TODO: Create a basic `Tensor`

use candle_core::{Device, Result, Tensor};

struct Model {
    t: Tensor,
}

impl Model {
    fn forward(&self, i: &Tensor) -> Result<Tensor> {
        // TODO: create a simple forward function here
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let t = Tensor::randn(0f32, 1.0, (8, 8), &device)?;

    // TODO: Create an instance of model

    let i = Tensor::randn(0f32, 1.0, (16, 8), &device)?;
    model.forward(&i)?;
    Ok(())
}
