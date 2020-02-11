#![feature(test)]
extern crate test;
use test::Bencher;

use futures::stream::StreamExt;
use futures_util::pin_mut;
use tokio::runtime;

use hidefix::{idx::Index, reader::DatasetReader, stream};

#[bench]
fn read_2d_chunked_idx(b: &mut Bencher) {
    let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
    let mut r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_2d_chunked_idx_stream(b: &mut Bencher) {
    let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
    let r =
        stream::DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    let mut rt = runtime::Runtime::new().unwrap();

    b.iter(|| {
        let vs: Vec<f32> = rt.block_on(async {
            let v = r.stream_values::<f32>(None, None);
            pin_mut!(v);
            v.map(|v| v.unwrap())
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .flatten()
                .collect()
        });

        vs
    })
}

#[bench]
fn read_2d_chunked_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/chunked_oneD.h5").unwrap();
    let d = h.dataset("d_4_chunks").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

#[bench]
fn read_t_float32_idx(b: &mut Bencher) {
    let i = Index::index("tests/data/t_float.h5").unwrap();
    let mut r = DatasetReader::with_dataset(i.dataset("d32_1").unwrap(), i.path()).unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_t_float32_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/t_float.h5").unwrap();
    let d = h.dataset("d32_1").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

#[bench]
fn read_chunked_1d_idx(b: &mut Bencher) {
    let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
    let mut r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_chunked_1d_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/chunked_oneD.h5").unwrap();
    let d = h.dataset("d_4_chunks").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

mod coads {
    use super::*;

    #[bench]
    fn native(b: &mut Bencher) {
        let h = hdf5::File::open("../data/coads_climatology.nc4").unwrap();
        let d = h.dataset("SST").unwrap();

        b.iter(|| d.read_raw::<f32>().unwrap())
    }

    #[bench]
    fn idx(b: &mut Bencher) {
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let mut r = DatasetReader::with_dataset(i.dataset("SST").unwrap(), i.path()).unwrap();

        b.iter(|| r.values::<f32>(None, None).unwrap())
    }
}

#[cfg(feature = "io_uring")]
mod uring {
    use hidefix::uring;

    #[bench]
    fn read_t_float32_idx_rio(b: &mut Bencher) {
        let i = Index::index("tests/data/t_float.h5").unwrap();
        let mut r =
            uring::DatasetReader::with_dataset(i.dataset("d32_1").unwrap(), i.path()).unwrap();

        b.iter(|| r.values::<f32>(None, None).unwrap())
    }
}
