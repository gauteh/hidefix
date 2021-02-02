#![feature(test)]
extern crate test;

use futures::executor::block_on_stream;
use futures::pin_mut;
use hidefix::prelude::*;

const SVIM: &'static str = "/home/gauteh/dev/dars/data/met/ocean_avg_19600101.nc4";

#[ignore]
#[test]
fn ocean_time() {
    use hidefix::idx::*;

    let h = hdf5::File::open(SVIM).unwrap();
    let d = h.dataset("ocean_time").unwrap();
    let hv = d.read_raw::<f64>().unwrap();

    let i = h.index().unwrap();
    let mut d = i.reader("ocean_time").unwrap();
    let v = d.values::<f64>(None, None).unwrap();

    assert_eq!(hv, v);

    let d = i.dataset("ocean_time").unwrap();
    if let DatasetD::D1(d) = d {
        println!("chunks: {:#?}", &d.chunks);
    } else {
        panic!("wrong dims");
    }

    let s = i.streamer("ocean_time").unwrap();
    let s = s.stream_values::<f64>(None, None);
    pin_mut!(s);
    let vs: Vec<f64> = block_on_stream(s).flatten().flatten().collect();

    assert_eq!(hv, vs);
}

#[ignore]
#[test]
fn temp() {
    use hidefix::idx::*;

    let h = hdf5::File::open(SVIM).unwrap();
    let d = h.dataset("temp").unwrap();
    let hv = d.read_raw::<i16>().unwrap();

    let i = h.index().unwrap();

    let d = i.dataset("temp").unwrap();
    if let DatasetD::D4(ds) = d {
        println!("chunks: {:#?}", &ds.chunks[..40]);
        println!("chunk shape: {:?}", ds.chunk_shape);
        println!("ds shape: {:?}", ds.shape);
    } else {
        panic!("wrong dims");
    }

    let mut d = i.reader("temp").unwrap();
    assert_eq!(d.dsize(), 2);

    let v = d.values::<i16>(None, None).unwrap();
    assert_eq!(hv, v);

    d.values::<i16>(Some(&[0, 0, 0, 0]), Some(&[1, 32, 580, 1202]))
        .unwrap();

    let s = i.streamer("temp").unwrap();
    let s = s.stream_values::<i16>(None, None);
    pin_mut!(s);
    let vs: Vec<i16> = block_on_stream(s).flatten().flatten().collect();
    assert_eq!(hv, vs);
}
