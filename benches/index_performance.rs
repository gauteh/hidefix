#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::idx::Index;

#[bench]
fn read_2d_chunked_idx(b: &mut Bencher) {
    b.iter(|| Index::index("tests/data/chunked_oneD.h5").unwrap())
}

mod coads {
    use super::*;

    #[bench]
    fn slicer(b: &mut Bencher) {
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();

        println!("slices: {}",
            d.chunk_slices(None, None).collect::<Vec<_>>().len());

        b.iter(|| d.chunk_slices(None, None).for_each(drop))
    }
}

