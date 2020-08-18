#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::idx::DatasetD;
use hidefix::idx::Index;

#[bench]
fn slicer(b: &mut Bencher) {
    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let d = i.dataset("SST").unwrap();
    if let DatasetD::D3(d) = d {
        b.iter(|| d.chunk_slices(None, None).for_each(drop))
    } else {
        panic!()
    }
}
