use candle_core::{Device, Tensor};
use candle_datasets::vision::mnist::load;
use candle_datasets::vision::Dataset;

use csv::ReaderBuilder;
use std::fs::File;
use std::io::BufReader;

#[allow(dead_code)]
struct CustomDataset {
    data: Tensor,
    target: Tensor,
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl CustomDataset {
    fn new(data: Tensor, target: Tensor) -> Self {
        Self { data, target }
    }
}

#[allow(unused_variables)]
fn main() {
    let data: Dataset = load().unwrap();

    let data_path = "data/mnist/data.csv";
    let target_path = "data/mnist/target.csv";

    let data_file = File::open(data_path).unwrap();
    let target_file = File::open(target_path).unwrap();

    let mut data_reader = ReaderBuilder::new().from_reader(BufReader::new(data_file));
    let mut target_reader = ReaderBuilder::new().from_reader(BufReader::new(target_file));

    let mut data: Vec<f32> = Vec::new();
    let mut target: Vec<f32> = Vec::new();

    data_reader.records().map(|x| x.unwrap()).for_each(|x| {
        x.iter()
            .map(|x| x.parse().unwrap())
            .for_each(|x| data.push(x));
    });

    target_reader.records().map(|x| x.unwrap()).for_each(|x| {
        x.iter()
            .map(|x| x.parse().unwrap())
            .for_each(|x| target.push(x));
    });

    let device = Device::Cpu;
    let data = Tensor::from_vec(data, (9999, 784), &device).unwrap();
    let target = Tensor::from_vec(target, (9999, 1), &device).unwrap();

    let dataset = CustomDataset { data, target };
}
