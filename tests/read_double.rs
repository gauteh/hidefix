use futures::executor::block_on_stream;
use futures::pin_mut;

use byte_slice_cast::AsMutSliceOf;
use hidefix::prelude::*;

#[test]
fn feb_nc4_double() {
    let i = Index::index("tests/data/feb.nc4").unwrap();
    let d = i.dataset("T").unwrap();
    println!("dataset: {:#?}", d);
    let mut r = i.reader("T").unwrap();

    let h = hdf5::File::open("tests/data/feb.nc4").unwrap();
    let d = h.dataset("T").unwrap();
    let hv = d.read_raw::<f64>().unwrap();

    let v = r.values::<f64>(None, None).unwrap();

    assert_eq!(hv, v);

    println!("{:?}", v);

    let r = i.streamer("T").unwrap();
    let s = r.stream_values::<f64>(None, None);
    pin_mut!(s);
    let vs: Vec<f64> = block_on_stream(s).flatten().flatten().collect();

    assert_eq!(hv, vs);
    println!("{:?}", vs);

    let sb = r.stream(None, None);
    pin_mut!(sb);
    let mut vb: Vec<u8> = block_on_stream(sb).flatten().flatten().collect();

    let vvb = vb.as_mut_slice_of::<f64>().unwrap();
    // vvb.to_native(Order::BE);
    assert_eq!(hv, vvb);
}
