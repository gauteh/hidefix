#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::idx::Index;

#[bench]
fn read_2d_chunked_idx(b: &mut Bencher) {
    b.iter(|| Index::index("tests/data/chunked_oneD.h5").unwrap())
}
