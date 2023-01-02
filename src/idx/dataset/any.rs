use std::path::Path;
use serde::{Deserialize, Serialize};

use crate::prelude::{Reader, Streamer};
use super::*;

/// Dataset in possible dimensions.
#[derive(Debug, Serialize, Deserialize)]
pub enum DatasetD<'a> {
    #[serde(borrow)]
    D0(Dataset<'a, 0>),
    #[serde(borrow)]
    D1(Dataset<'a, 1>),
    #[serde(borrow)]
    D2(Dataset<'a, 2>),
    #[serde(borrow)]
    D3(Dataset<'a, 3>),
    #[serde(borrow)]
    D4(Dataset<'a, 4>),
    #[serde(borrow)]
    D5(Dataset<'a, 5>),
    #[serde(borrow)]
    D6(Dataset<'a, 6>),
    #[serde(borrow)]
    D7(Dataset<'a, 7>),
    #[serde(borrow)]
    D8(Dataset<'a, 8>),
    #[serde(borrow)]
    D9(Dataset<'a, 9>),
}

impl DatasetD<'_> {
    pub fn index(ds: &hdf5::Dataset) -> Result<DatasetD<'static>, anyhow::Error> {
        use DatasetD::*;

        match ds.ndim() {
            0 => Ok(D0(Dataset::<0>::index(ds)?)),
            1 => Ok(D1(Dataset::<1>::index(ds)?)),
            2 => Ok(D2(Dataset::<2>::index(ds)?)),
            3 => Ok(D3(Dataset::<3>::index(ds)?)),
            4 => Ok(D4(Dataset::<4>::index(ds)?)),
            5 => Ok(D5(Dataset::<5>::index(ds)?)),
            6 => Ok(D6(Dataset::<6>::index(ds)?)),
            7 => Ok(D7(Dataset::<7>::index(ds)?)),
            8 => Ok(D8(Dataset::<8>::index(ds)?)),
            9 => Ok(D9(Dataset::<9>::index(ds)?)),
            n => panic!("Dataset only implemented for 0..9 dimensions (not {})", n),
        }
    }

    pub fn as_reader(&self, path: &Path) -> Result<Box<dyn Reader + '_>, anyhow::Error> {
        use crate::reader::cache::CacheReader;
        use std::fs;
        use DatasetD::*;

        Ok(match self {
            D0(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D1(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D2(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D3(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D4(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D5(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D6(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D7(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D8(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
            D9(ds) => Box::new(CacheReader::with_dataset(ds, fs::File::open(path)?)?),
        })
    }

    pub fn as_streamer(&self, path: &Path) -> Result<Box<dyn Streamer + '_>, anyhow::Error> {
        use crate::reader::stream::StreamReader;
        use DatasetD::*;

        Ok(match self {
            D0(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D1(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D2(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D3(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D4(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D5(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D6(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D7(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D8(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
            D9(ds) => Box::new(StreamReader::with_dataset(ds, path)?),
        })
    }

    pub fn dsize(&self) -> usize {
        use DatasetD::*;
        match self {
            D0(ds) => ds.dsize,
            D1(ds) => ds.dsize,
            D2(ds) => ds.dsize,
            D3(ds) => ds.dsize,
            D4(ds) => ds.dsize,
            D5(ds) => ds.dsize,
            D6(ds) => ds.dsize,
            D7(ds) => ds.dsize,
            D8(ds) => ds.dsize,
            D9(ds) => ds.dsize,
        }
    }

    pub fn dtype(&self) -> Datatype {
        use DatasetD::*;
        match self {
            D0(ds) => ds.dtype,
            D1(ds) => ds.dtype,
            D2(ds) => ds.dtype,
            D3(ds) => ds.dtype,
            D4(ds) => ds.dtype,
            D5(ds) => ds.dtype,
            D6(ds) => ds.dtype,
            D7(ds) => ds.dtype,
            D8(ds) => ds.dtype,
            D9(ds) => ds.dtype,
        }
    }
}
