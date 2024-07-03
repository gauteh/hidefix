use divan::Bencher;

use hidefix::idx::DatasetD;
use hidefix::idx::Index;

#[divan::bench]
fn slicer(b: Bencher) {
    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let d = i.dataset("SST").unwrap();
    if let DatasetD::D3(d) = d {
        b.bench_local(|| d.chunk_slices(..).for_each(drop))
    } else {
        panic!()
    }
}

fn main() {
    divan::main();
}
