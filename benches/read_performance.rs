#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::{idx::Index, reader::DatasetReader};

#[bench]
fn read_2d_chunked_idx(b: &mut Bencher) {
    let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
    let mut r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
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

#[cfg(feature = "io_uring")]
mod uring {
    use hidefix::uring;

    #[bench]
    fn read_t_float32_idx_rio(b: &mut Bencher) {
        let i = Index::index("tests/data/t_float.h5").unwrap();
        let mut r = uring::DatasetReader::with_dataset(i.dataset("d32_1").unwrap(), i.path()).unwrap();

        b.iter(|| r.values::<f32>(None, None).unwrap())
    }
}
