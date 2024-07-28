#[allow(unused_imports)]
use candle_core::{Device, Tensor};
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
    // TODO: Use the mnist datasets load function
    let data: Dataset;

    // TODO: Load from local files
    let data_path = "../../data/mnist/data.csv";
    let target_path = "../../data/mnist/target.csv";

    let data_file = File::open(data_path).unwrap();
    let target_file = File::open(target_path).unwrap();

    let data_reader = ReaderBuilder::new().from_reader(BufReader::new(data_file));
    let target_reader = ReaderBuilder::new().from_reader(BufReader::new(target_file));

    let dataset = CustomDataset { data, target };
}
