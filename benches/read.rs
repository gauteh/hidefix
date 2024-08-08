use divan::Bencher;
use hidefix::prelude::*;

#[divan::bench]
fn read_2d_chunked(b: Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let mut r = i.reader("d_4_chunks").unwrap();

    b.bench_local(|| r.values::<f32, _>(..).unwrap())
}

#[divan::bench]
fn read_2d_shuffled(b: Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_shuffled_twoD.h5").unwrap();
    let mut r = i.reader("d_4_shuffled_chunks").unwrap();

    b.bench_local(|| r.values::<f32, _>(..).unwrap())
}

#[divan::bench]
fn read_2d_compressed(b: Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_gzipped_twoD.h5").unwrap();
    let mut r = i.reader("d_4_gzipped_chunks").unwrap();

    b.bench_local(|| r.values::<f32, _>(..).unwrap())
}

#[divan::bench]
fn read_2d_shuffled_compressed(b: Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
    let mut r = i.reader("d_4_shufzip_chunks").unwrap();

    b.bench_local(|| r.values::<f32, _>(..).unwrap())
}

#[divan::bench]
fn read_t_float32(b: Bencher) {
    let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();
    let mut r = i.reader("d32_1").unwrap();

    b.bench_local(|| r.values::<f32, _>(..).unwrap())
}

#[divan::bench]
fn read_chunked_1d(b: Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let mut r = i.reader("d_4_chunks").unwrap();

    b.bench_local(|| r.values::<f32, _>(..).unwrap())
}

#[divan::bench]
fn coads(b: Bencher) {
    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let mut r = i.reader("SST").unwrap();

    {
        let h = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
        let d = h.dataset("SST").unwrap();

        assert_eq!(
            d.read_raw::<f32>().unwrap(),
            r.values::<f32, _>(..).unwrap()
        );
    }

    b.bench_local(|| r.values::<f32, _>(..).unwrap())
}

fn main() {
    divan::main();
}
