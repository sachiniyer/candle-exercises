use candle_core::{Device, Tensor};

#[allow(unused_variables)]
fn main() {
    let device = Device::Cpu;
    let tensor = Tensor::randn(0f32, 1.0, (8, 8), &device);
}
