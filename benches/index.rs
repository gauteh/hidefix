#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::idx::Index;

#[bench]
fn chunked_1d(b: &mut Bencher) {
    b.iter(|| Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap())
}

#[ignore]
#[bench]
fn meps(b: &mut Bencher) {
    b.iter(|| test::black_box(Index::index("tests/data/meps_det_vc_2_5km_latest.nc")).unwrap())
}
