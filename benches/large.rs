#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::prelude::*;
use ndarray::s;

const FILE: &'static str = env!("HIDEFIX_LARGE_FILE");
const VAR: &'static str = env!("HIDEFIX_LARGE_VAR");

#[ignore]
#[bench]
fn idx_small_slice(b: &mut Bencher) {
    let i = Index::index(FILE).unwrap();
    let mut r = i.reader(VAR).unwrap();

    // test against native
    let h = hdf5::File::open(FILE).unwrap();
    let d = h.dataset(VAR).unwrap();
    let hv = d
        .read_slice_1d::<i32, _>(s![0..2, 0..2, 0..1, 0..5])
        .unwrap()
        .iter()
        .map(|v| *v)
        .collect::<Vec<i32>>();

    assert_eq!(
        hv,
        r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[2, 2, 1, 5]))
            .unwrap()
    );

    b.iter(|| {
        test::black_box(
            r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[2, 2, 1, 5]))
                .unwrap(),
        )
    });
}

#[ignore]
#[bench]
fn native_small_slice(b: &mut Bencher) {
    let h = hdf5::File::open(FILE).unwrap();
    let d = h.dataset(VAR).unwrap();

    b.iter(|| {
        test::black_box(
            d.read_slice_1d::<i32, _>(s![0..2, 0..2, 0..1, 0..5])
                .unwrap(),
        )
    })
}

#[ignore]
#[bench]
fn idx_med_slice(b: &mut Bencher) {
    let i = Index::index(FILE).unwrap();
    let mut r = i.reader(VAR).unwrap();

    // test against native
    let h = hdf5::File::open(FILE).unwrap();
    let d = h.dataset(VAR).unwrap();
    let hv = d
        .read_slice_1d::<i32, _>(s![0..10, 0..10, 0..1, 0..700])
        .unwrap()
        .iter()
        .map(|v| *v)
        .collect::<Vec<i32>>();

    assert_eq!(
        hv,
        r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[10, 10, 1, 700]))
            .unwrap()
    );

    b.iter(|| {
        test::black_box(
            r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[10, 10, 1, 700]))
                .unwrap(),
        )
    });
}

#[ignore]
#[bench]
fn native_med_slice(b: &mut Bencher) {
    let h = hdf5::File::open(FILE).unwrap();
    let d = h.dataset(VAR).unwrap();

    b.iter(|| {
        test::black_box(
            d.read_slice_1d::<i32, _>(s![0..10, 0..10, 0..1, 0..20000])
                .unwrap(),
        )
    })
}

#[ignore]
#[bench]
fn idx_big_slice(b: &mut Bencher) {
    let i = Index::index(FILE).unwrap();
    let mut r = i.reader(VAR).unwrap();

    // test against native
    let h = hdf5::File::open(FILE).unwrap();
    let d = h.dataset(VAR).unwrap();
    let hv = d
        .read_slice_1d::<i32, _>(s![0..24, 0..16, 0..1, 0..739])
        .unwrap()
        .iter()
        .map(|v| *v)
        .collect::<Vec<i32>>();

    assert_eq!(
        hv,
        r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[24, 16, 1, 739]))
            .unwrap()
    );

    b.iter(|| {
        test::black_box(
            r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[24, 16, 1, 739]))
                .unwrap(),
        )
    });
}

#[ignore]
#[bench]
fn native_big_slice(b: &mut Bencher) {
    let h = hdf5::File::open(FILE).unwrap();
    let d = h.dataset(VAR).unwrap();

    b.iter(|| {
        test::black_box(
            d.read_slice_1d::<i32, _>(s![0..65, 0..65, 0..1, 0..20000])
                .unwrap(),
        )
    })
}
