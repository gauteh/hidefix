use divan::Bencher;

#[divan::bench]
fn read_2d_shuffled(b: Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_shuffled_twoD.h5").unwrap();
    let d = h.dataset("d_4_shuffled_chunks").unwrap();

    b.bench_local(|| d.read_raw::<f32>().unwrap())
}

#[divan::bench]
fn read_2d_compressed(b: Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_gzipped_twoD.h5").unwrap();
    let d = h.dataset("d_4_gzipped_chunks").unwrap();

    b.bench_local(|| d.read_raw::<f32>().unwrap())
}

#[divan::bench]
fn read_2d_shuffled_compressed(b: Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
    let d = h.dataset("d_4_shufzip_chunks").unwrap();

    b.bench_local(|| d.read_raw::<f32>().unwrap())
}

#[divan::bench]
fn read_2d_chunked(b: Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let d = h.dataset("d_4_chunks").unwrap();

    b.bench_local(|| d.read_raw::<f32>().unwrap())
}

#[divan::bench]
fn read_t_float32(b: Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/t_float.h5").unwrap();
    let d = h.dataset("d32_1").unwrap();

    b.bench_local(|| d.read_raw::<f32>().unwrap())
}

#[divan::bench]
fn read_chunked_1d(b: Bencher) {
    let h = hdf5::File::open("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let d = h.dataset("d_4_chunks").unwrap();

    b.bench_local(|| d.read_raw::<f32>().unwrap())
}

#[divan::bench]
fn coads_values(b: Bencher) {
    let h = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
    let d = h.dataset("SST").unwrap();

    b.bench_local(|| d.read_raw::<f32>().unwrap())
}

fn main() {
    divan::main();
}
