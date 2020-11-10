#![feature(test)]
extern crate test;

use hidefix::prelude::*;
use ndarray::s;

/// Test whether datasets of various rank are correctly sliced.

#[test]
fn chunked_2d() {
    type T = f32;

    let i = Index::index("tests/data/dmrpp/chunked_twoD.h5").unwrap();
    let mut r = i.reader("d_4_chunks").unwrap();

    let values = r.values::<T>(None, None).unwrap();

    let h = hdf5::File::open("tests/data/dmrpp/chunked_twoD.h5").unwrap();
    let d = h.dataset("d_4_chunks").unwrap();
    let hval = d.read_raw::<T>().unwrap();

    assert_eq!(values, hval);

    let values = r.values::<T>(Some(&[10, 10]), Some(&[15, 15])).unwrap();
    let hvs = d.read_dyn::<T>().unwrap();
    let hval = hvs
        .slice(s![10..25, 10..25])
        .iter()
        .map(|v| *v)
        .collect::<Vec<T>>();
    assert_eq!(values, hval);
}

#[test]
fn chunked_3d() {
    type T = f32;

    let i = Index::index("tests/data/dmrpp/chunked_threeD.h5").unwrap();
    let mut r = i.reader("d_8_chunks").unwrap();

    let values = r.values::<T>(None, None).unwrap();

    let h = hdf5::File::open("tests/data/dmrpp/chunked_threeD.h5").unwrap();
    let d = h.dataset("d_8_chunks").unwrap();
    let hval = d.read_raw::<T>().unwrap();

    assert_eq!(values, hval);

    let values = r
        .values::<T>(Some(&[10, 10, 10]), Some(&[1, 2, 1]))
        .unwrap();
    let hvs = d.read_dyn::<T>().unwrap();
    let hval = hvs
        .slice(s![10..11, 10..12, 10..11])
        .iter()
        .map(|v| *v)
        .collect::<Vec<T>>();
    assert_eq!(values, hval);
}

#[test]
fn chunked_4d() {
    type T = f32;

    let i = Index::index("tests/data/dmrpp/chunked_fourD.h5").unwrap();
    let mut r = i.reader("d_16_chunks").unwrap();

    let values = r.values::<T>(None, None).unwrap();

    let h = hdf5::File::open("tests/data/dmrpp/chunked_fourD.h5").unwrap();
    let d = h.dataset("d_16_chunks").unwrap();
    let hval = d.read_raw::<T>().unwrap();

    assert_eq!(values, hval);

    let values = r
        .values::<T>(Some(&[10, 10, 10, 5]), Some(&[15, 15, 15, 14]))
        .unwrap();
    let hvs = d.read_dyn::<T>().unwrap();
    let hval = hvs
        .slice(s![10..25, 10..25, 10..25, 5..19])
        .iter()
        .map(|v| *v)
        .collect::<Vec<T>>();
    assert_eq!(values, hval);
}

#[test]
fn scalar() {
    type T = i32;

    let i = Index::index("tests/data/dmrpp/t_int_scalar.h5").unwrap();
    let mut r = i.reader("scalar").unwrap();

    let values = r.values::<T>(None, None).unwrap();

    let h = hdf5::File::open("tests/data/dmrpp/t_int_scalar.h5").unwrap();
    let d = h.dataset("scalar").unwrap();
    let hval = d.read_raw::<T>().unwrap();

    assert_eq!(values, hval);
}
