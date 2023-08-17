use crate::filters::byteorder::Order;
use std::fs::File;
use std::path::{Path, PathBuf};

use super::{
    chunk::{decode_chunk, read_chunk, read_chunk_to},
    dataset::{ParReader, Reader},
};
use crate::idx::{Chunk, Dataset};

pub struct S3Reader<'a, const D: usize> {
    ds: &'a Dataset<'a, D>,
    path: PathBuf,
    chunk_sz: u64,
}

impl<'a, const D: usize> S3Reader<'a, D> {
    pub fn with_dataset<P: AsRef<Path>>(
        ds: &'a Dataset<D>,
        path: P,
    ) -> Result<S3Reader<'a, D>, anyhow::Error> {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(S3Reader {
            ds,
            path: path.as_ref().into(),
            chunk_sz,
        })
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn terrafusion() {
        let url = "s3://terrafusiondatasampler/P233/TERRA_BF_L1B_O12236_20020406135439_F000_V001.h5";
    }
}
