#![feature(test)]
extern crate test;
use test::Bencher;

use futures::executor::block_on_stream;
use futures::pin_mut;

use hidefix::{
    idx::Index,
    reader::{cache, simple, stream},
};

#[bench]
fn read_2d_chunked_idx(b: &mut Bencher) {
    let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
    let mut r =
        simple::DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_2d_chunked_cache(b: &mut Bencher) {
    let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
    let mut r =
        cache::DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_2d_chunked_idx_stream(b: &mut Bencher) {
    let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
    let r =
        stream::DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    b.iter(|| {
        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        block_on_stream(v).for_each(drop);
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
    let mut r = simple::DatasetReader::with_dataset(i.dataset("d32_1").unwrap(), i.path()).unwrap();

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
    let mut r =
        simple::DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

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
        let mut r =
            simple::DatasetReader::with_dataset(i.dataset("SST").unwrap(), i.path()).unwrap();

        {
            let h = hdf5::File::open("../data/coads_climatology.nc4").unwrap();
            let d = h.dataset("SST").unwrap();

            assert_eq!(
                d.read_raw::<f32>().unwrap(),
                r.values::<f32>(None, None).unwrap()
            );
        }

        b.iter(|| r.values::<f32>(None, None).unwrap())
    }

    #[bench]
    fn chunk_at_coord(b: &mut Bencher) {
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();

        b.iter(|| d.chunk_at_coord(&[5, 15, 40]))
    }

    #[bench]
    fn chunk_slices_range(b: &mut Bencher) {
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();

        b.iter(|| d.chunk_slices(None, None).for_each(drop));
    }

    #[bench]
    fn cache(b: &mut Bencher) {
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let mut r =
            cache::DatasetReader::with_dataset(i.dataset("SST").unwrap(), i.path()).unwrap();

        {
            let h = hdf5::File::open("../data/coads_climatology.nc4").unwrap();
            let d = h.dataset("SST").unwrap();

            assert_eq!(
                d.read_raw::<f32>().unwrap(),
                r.values::<f32>(None, None).unwrap()
            );
        }

        b.iter(|| r.values::<f32>(None, None).unwrap())
    }

    #[bench]
    fn stream(b: &mut Bencher) {
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let r = stream::DatasetReader::with_dataset(i.dataset("SST").unwrap(), i.path()).unwrap();

        {
            let v = r.stream_values::<f32>(None, None);
            pin_mut!(v);
            let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

            let h = hdf5::File::open("../data/coads_climatology.nc4").unwrap();
            let d = h.dataset("SST").unwrap();
            assert_eq!(d.read_raw::<f32>().unwrap(), vs);
        }

        b.iter(|| {
            let v = r.stream_values::<f32>(None, None);
            pin_mut!(v);
            block_on_stream(v).for_each(drop);
        })
    }

    #[bench]
    fn stream_bytes(b: &mut Bencher) {
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let r = stream::DatasetReader::with_dataset(i.dataset("SST").unwrap(), i.path()).unwrap();

        use futures::executor::block_on_stream;

        b.iter(|| {
            let v = r.stream(None, None);
            pin_mut!(v);
            block_on_stream(v).for_each(drop);
        })
    }

    #[bench]
    fn idx_bytes(b: &mut Bencher) {
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let mut r =
            simple::DatasetReader::with_dataset(i.dataset("SST").unwrap(), i.path()).unwrap();

        b.iter(|| r.read(None, None).unwrap())
    }
}

#[cfg(feature = "io_uring")]
mod uring {
    use super::*;
    use hidefix::reader::uring;

    #[bench]
    fn read_t_float32_idx_rio(b: &mut Bencher) {
        let i = Index::index("tests/data/t_float.h5").unwrap();

        b.iter(|| {
            let r =
                uring::DatasetReader::with_dataset(i.dataset("d32_1").unwrap(), i.path()).unwrap();
            r.values::<f32>(None, None).unwrap();

            // use std::{thread, time};
            // let ten_millis = time::Duration::from_millis(1);
            // thread::sleep(ten_millis);
        })
    }
}
