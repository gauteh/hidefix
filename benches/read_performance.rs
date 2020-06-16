#![feature(test)]
extern crate test;
use test::Bencher;

use futures::executor::block_on_stream;
use futures::pin_mut;

use hidefix::idx::Index;

#[bench]
fn read_2d_chunked_cache(b: &mut Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let mut r = i.reader("d_4_chunks").unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_2d_shuffled_cache(b: &mut Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_shuffled_twoD.h5").unwrap();
    let mut r = i.reader("d_4_shuffled_chunks").unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_2d_shuffled_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_shuffled_twoD.h5").unwrap();
    let d = h.dataset("d_4_shuffled_chunks").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

#[bench]
fn read_2d_compressed_cache(b: &mut Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_gzipped_twoD.h5").unwrap();
    let mut r = i.reader("d_4_gzipped_chunks").unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_2d_compressed_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_gzipped_twoD.h5").unwrap();
    let d = h.dataset("d_4_gzipped_chunks").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

#[bench]
fn read_2d_shuffled_compressed_cache(b: &mut Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
    let mut r = i.reader("d_4_shufzip_chunks").unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_2d_shuffled_compressed_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
    let d = h.dataset("d_4_shufzip_chunks").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

#[bench]
fn read_2d_chunked_stream(b: &mut Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let r = i.streamer("d_4_chunks").unwrap();

    b.iter(|| {
        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        block_on_stream(v).for_each(drop);
    })
}

#[bench]
fn read_2d_chunked_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let d = h.dataset("d_4_chunks").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

#[bench]
fn read_t_float32_cache(b: &mut Bencher) {
    let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();
    let mut r = i.reader("d32_1").unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_t_float32_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/t_float.h5").unwrap();
    let d = h.dataset("d32_1").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

#[bench]
fn read_chunked_1d_cache(b: &mut Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let mut r = i.reader("d_4_chunks").unwrap();

    b.iter(|| r.values::<f32>(None, None).unwrap())
}

#[bench]
fn read_chunked_1d_nat(b: &mut Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let d = h.dataset("d_4_chunks").unwrap();

    b.iter(|| d.read_raw::<f32>().unwrap())
}

mod coads {
    use super::*;

    #[bench]
    fn native(b: &mut Bencher) {
        let h = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
        let d = h.dataset("SST").unwrap();

        b.iter(|| d.read_raw::<f32>().unwrap())
    }

    #[bench]
    fn chunk_at_coord(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();

        b.iter(|| d.chunk_at_coord(&[5, 15, 40]))
    }

    #[bench]
    fn chunk_slices_range(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();

        b.iter(|| d.chunk_slices(None, None).for_each(drop));
    }

    #[bench]
    fn cache(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let mut r = i.reader("SST").unwrap();

        {
            let h = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
            let d = h.dataset("SST").unwrap();

            assert_eq!(
                d.read_raw::<f32>().unwrap(),
                r.values::<f32>(None, None).unwrap()
            );
        }

        b.iter(|| r.values::<f32>(None, None).unwrap())
    }

    #[bench]
    fn async_cache(b: &mut Bencher) {
        use futures::executor::block_on;
        use futures::io::AsyncReadExt;
        use async_std::fs;
        use byte_slice_cast::IntoVecOf;
        use hidefix::filters::byteorder::ToNative;
        use hidefix::reader::async_cache;

        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let ds = i.dataset("SST").unwrap();

        b.iter(|| block_on(async {
            let fd = fs::File::open(i.path().unwrap()).await.unwrap();
            let mut r = async_cache::DatasetReader::with_dataset_read(ds, fd, None, None).unwrap();
            let mut buf = Vec::with_capacity(r.size());

            // let mut r = BufReader::new(r);

            r.read_to_end(&mut buf).await.unwrap();
            let mut vs = buf.into_vec_of::<f32>().unwrap();
            vs.to_native(ds.order)
        }))
    }

    #[bench]
    fn stream(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let r = i.streamer("SST").unwrap();

        {
            let v = r.stream_values::<f32>(None, None);
            pin_mut!(v);
            let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

            let h = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
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
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let r = i.streamer("SST").unwrap();

        b.iter(|| {
            let v = r.stream(None, None);
            pin_mut!(v);
            block_on_stream(v).for_each(drop);
        })
    }
}

#[cfg(feature = "io_uring")]
mod uring {
    use super::*;
    use hidefix::reader::uring;

    #[bench]
    fn read_t_float32_idx_rio(b: &mut Bencher) {
        let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();

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
