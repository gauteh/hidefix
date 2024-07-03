use divan::Bencher;

use hidefix::idx::Index;

#[divan::bench]
fn chunked_1d(b: Bencher) {
    b.bench_local(|| Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap())
}

fn main() {
    divan::main();
}
