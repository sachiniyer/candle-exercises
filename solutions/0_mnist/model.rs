use candle_core::{Device, Result, Tensor};

struct Model {
    t: Tensor,
}

impl Model {
    fn forward(&self, i: &Tensor) -> Result<Tensor> {
        Ok(i.matmul(&self.t)?)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let t = Tensor::randn(0f32, 1.0, (8, 8), &device)?;
    let model = Model { t };
    let i = Tensor::randn(0f32, 1.0, (16, 8), &device)?;
    model.forward(&i)?;
    Ok(())
}
