#![feature(test)]
extern crate test;
use test::Bencher;

use std::path::PathBuf;
use std::sync::Mutex;

use hidefix::prelude::*;
use ndarray::{s, IxDyn};

const URL: &'static str = "https://thredds.met.no/thredds/fileServer/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.2023081600.nc";
const VAR: &'static str = "u_eastward";

type T = f32;

fn get_file() -> PathBuf {
    use std::time::Duration;

    static NK: Mutex<()> = Mutex::new(());
    let _guard = NK.lock().unwrap();

    let mut p = std::env::temp_dir();
    p.push("hidefix");

    let d = p.clone();

    p.push("norkyst.nc");

    if !p.exists() {
        println!("downloading norkyst file to {p:#?}..");
        std::fs::create_dir_all(&d).unwrap();
        let c = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10 * 60))
            .build()
            .unwrap();
        let r = c.get(URL).send().unwrap();
        std::fs::write(&p, r.bytes().unwrap()).unwrap();
    }

    p
}

#[ignore]
#[bench]
fn idx_small_slice(b: &mut Bencher) {
    let p = get_file();
    let i = Index::index(&p).unwrap();
    let mut r = i.reader(VAR).unwrap();

    // test against native
    let h = hdf5::File::open(&p).unwrap();
    let d = h.dataset(VAR).unwrap();
    let hv = d
        .read_slice::<T, _, IxDyn>(s![0..2, 0..2, 0..1, 0..5])
        .unwrap()
        .iter()
        .map(|v| *v)
        .collect::<Vec<T>>();

    assert_eq!(
        hv,
        r.values::<T>(Some(&[0, 0, 0, 0]), Some(&[2, 2, 1, 5]))
            .unwrap()
    );

    b.iter(|| {
        test::black_box(
            r.values::<T>(Some(&[0, 0, 0, 0]), Some(&[2, 2, 1, 5]))
                .unwrap(),
        )
    });
}

#[ignore]
#[bench]
fn native_small_slice(b: &mut Bencher) {
    let p = get_file();
    let h = hdf5::File::open(&p).unwrap();
    let d = h.dataset(VAR).unwrap();

    b.iter(|| {
        test::black_box(
            d.read_slice::<T, _, IxDyn>(s![0..2, 0..2, 0..1, 0..5])
                .unwrap(),
        )
    })
}

#[ignore]
#[bench]
fn idx_med_slice(b: &mut Bencher) {
    let p = get_file();
    let i = Index::index(&p).unwrap();
    let mut r = i.reader(VAR).unwrap();

    // test against native
    let h = hdf5::File::open(&p).unwrap();
    let d = h.dataset(VAR).unwrap();
    let hv = d
        .read_slice::<T, _, IxDyn>(s![0..10, 0..10, 0..1, 0..700])
        .unwrap()
        .iter()
        .map(|v| *v)
        .collect::<Vec<T>>();

    assert_eq!(
        hv,
        r.values::<T>(Some(&[0, 0, 0, 0]), Some(&[10, 10, 1, 700]))
            .unwrap()
    );

    b.iter(|| {
        test::black_box(
            r.values::<T>(Some(&[0, 0, 0, 0]), Some(&[10, 10, 1, 2602]))
                .unwrap(),
        )
    });
}

#[ignore]
#[bench]
fn native_med_slice(b: &mut Bencher) {
    let p = get_file();
    let h = hdf5::File::open(&p).unwrap();
    let d = h.dataset(VAR).unwrap();

    b.iter(|| {
        test::black_box(
            d.read_slice::<T, _, IxDyn>(s![0..10, 0..10, 0..1, 0..2602])
                .unwrap(),
        )
    })
}

#[ignore]
#[bench]
fn idx_big_slice(b: &mut Bencher) {
    let p = get_file();
    let i = Index::index(&p).unwrap();
    let mut r = i.reader(VAR).unwrap();

    // test against native
    let h = hdf5::File::open(&p).unwrap();
    let d = h.dataset(VAR).unwrap();
    let hv = d
        .read_slice::<T, _, IxDyn>(s![0..24, 0..16, 0..1, 0..739])
        .unwrap()
        .iter()
        .map(|v| *v)
        .collect::<Vec<T>>();

    assert_eq!(
        hv,
        r.values::<T>(Some(&[0, 0, 0, 0]), Some(&[24, 16, 1, 739]))
            .unwrap()
    );

    b.iter(|| {
        test::black_box(
            r.values::<T>(Some(&[0, 0, 0, 0]), Some(&[24, 16, 1, 2602]))
                .unwrap(),
        )
    });
}

#[ignore]
#[bench]
fn native_big_slice(b: &mut Bencher) {
    let p = get_file();
    let h = hdf5::File::open(&p).unwrap();
    let d = h.dataset(VAR).unwrap();

    b.iter(|| {
        test::black_box(
            d.read_slice::<T, _, IxDyn>(s![0..24, 0..16, 0..1, 0..2602])
                .unwrap(),
        )
    })
}
