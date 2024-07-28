#![allow(dead_code)]
#![allow(unused_variables)]

use candle_core::{Device, Tensor};
use candle_datasets::vision::mnist::load;
use candle_datasets::vision::Dataset;

use csv::ReaderBuilder;
use std::fs::File;
use std::io::BufReader;

use rand::seq::SliceRandom;
use rand::thread_rng;

struct CustomDataset {
    data: Tensor,
    target: Tensor,
}

impl CustomDataset {
    fn new(data: Tensor, target: Tensor) -> Self {
        Self { data, target }
    }

    fn len(&self) -> usize {
        self.data.dim(0).unwrap()
    }

    fn get(&self, index: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let data = self.data.get(index)?;
        let target = self.target.get(index)?;
        Ok((data, target))
    }
}

struct DataLoader {
    dataset: CustomDataset,
    batch_size: usize,
    shuffle: bool,
}

impl DataLoader {
    fn new(dataset: CustomDataset, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
        }
    }

    fn iter(&self) -> DataLoaderIterator {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            indices.shuffle(&mut thread_rng());
        }
        DataLoaderIterator {
            dataloader: self,
            indices,
            current_index: 0,
        }
    }
}

struct DataLoaderIterator<'a> {
    dataloader: &'a DataLoader,
    indices: Vec<usize>,
    current_index: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = candle_core::Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.indices.len() {
            return None;
        }

        let batch_end = (self.current_index + self.dataloader.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_index..batch_end];

        let mut batch_data = Vec::new();
        let mut batch_targets = Vec::new();

        for &idx in batch_indices {
            match self.dataloader.dataset.get(idx) {
                Ok((data, target)) => {
                    batch_data.push(data);
                    batch_targets.push(target);
                }
                Err(e) => return Some(Err(e)),
            }
        }

        self.current_index = batch_end;

        match (
            Tensor::stack(&batch_data, 0),
            Tensor::stack(&batch_targets, 0),
        ) {
            (Ok(data), Ok(targets)) => Some(Ok((data, targets))),
            (Err(e), _) | (_, Err(e)) => Some(Err(e)),
        }
    }
}

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
    let data = Tensor::from_vec(data, (99, 784), &device).unwrap();
    let target = Tensor::from_vec(target, (99, 1), &device).unwrap();

    let dataset = CustomDataset { data, target };
}
