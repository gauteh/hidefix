#![feature(test)]
#![allow(non_snake_case)]
extern crate test;

use hidefix::idx::{Dataset, DatasetD};
use hidefix::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;

const URL: &'static str = "https://thredds.met.no/thredds/fileServer/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.2023081600.nc";

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
#[test]
fn coords() {
    let p = get_file();

    let h = hdf5::File::open(&p).unwrap();
    let Y = h.dataset("Y").unwrap().read_raw::<f64>().unwrap();
    let X = h.dataset("X").unwrap().read_raw::<f64>().unwrap();

    let hi = Index::index(&p).unwrap();
    let hY = hi.reader("Y").unwrap().values::<f64>(None, None).unwrap();
    let hX = hi.reader("X").unwrap().values::<f64>(None, None).unwrap();

    assert_eq!(Y, hY);
    assert_eq!(X, hX);
}

#[ignore]
#[test]
fn wind() {
    let p = get_file();

    let h = hdf5::File::open(&p).unwrap();
    let Uw = h.dataset("Uwind").unwrap().read_raw::<f32>().unwrap();
    let Vw = h.dataset("Vwind").unwrap().read_raw::<f32>().unwrap();

    let hi = Index::index(&p).unwrap();
    let hUw = hi
        .reader("Uwind")
        .unwrap()
        .values::<f32>(None, None)
        .unwrap();
    let hVw = hi
        .reader("Vwind")
        .unwrap()
        .values::<f32>(None, None)
        .unwrap();

    hi.dataset("Uwind").unwrap().valid().unwrap();

    assert_eq!(Uw, hUw);
    assert_eq!(Vw, hVw);
}

#[ignore]
#[test]
fn current() {
    let p = get_file();

    let h = hdf5::File::open(&p).unwrap();
    let u = h.dataset("u_eastward").unwrap().read_raw::<f32>().unwrap();
    let v = h.dataset("v_northward").unwrap().read_raw::<f32>().unwrap();

    assert_eq!(u.len(), h.dataset("u_eastward").unwrap().size());

    let hi = Index::index(&p).unwrap();

    assert_eq!(u.len(), hi.dataset("u_eastward").unwrap().size());

    // hi.dataset("u_eastward").unwrap().valid().unwrap();

    let hu = hi
        .reader("u_eastward")
        .unwrap()
        .values::<f32>(None, None)
        .unwrap();
    let hv = hi
        .reader("v_northward")
        .unwrap()
        .values::<f32>(None, None)
        .unwrap();

    assert_eq!(u, hu);
    assert_eq!(v, hv);
}

#[ignore]
#[test]
fn temperature_salinity() {
    let p = get_file();

    let h = hdf5::File::open(&p).unwrap();
    let Uw = h.dataset("temperature").unwrap().read_raw::<i16>().unwrap();
    let Vw = h.dataset("salinity").unwrap().read_raw::<i16>().unwrap();

    let hi = Index::index(&p).unwrap();
    let hUw = hi
        .reader("temperature")
        .unwrap()
        .values::<i16>(None, None)
        .unwrap();
    let hVw = hi
        .reader("salinity")
        .unwrap()
        .values::<i16>(None, None)
        .unwrap();

    assert_eq!(Uw, hUw);
    assert_eq!(Vw, hVw);
}

#[ignore]
#[test]
fn chunk_slice_fracture() {
    let p = get_file();
    let hi = Index::index(&p).unwrap();

    // test that chunks are not unnecessarily fractured
    fn test_slices<const D: usize>(ds: &Dataset<D>) {
        let chunks = ds.chunk_slices(None, None).collect::<Vec<_>>();

        println!("chunks len: {}", chunks.len());

        // might have to make `chunks` unique.
        assert_eq!(chunks.len(), ds.chunks.len());

        for i in 1..chunks.len() {
            let p = chunks[i - 1];
            let c = chunks[i];

            // assert_eq!(p.2 - p.1, chunk_total_size);
            // assert_eq!(c.2 - c.1, chunk_total_size);

            assert_ne!(c, p);
        }
    }

    let DatasetD::D1(ds) = hi.dataset("X").unwrap() else {
        panic!("wrong dims")
    };
    test_slices(ds);

    let DatasetD::D4(ds) = hi.dataset("temperature").unwrap() else {
        panic!("wrong dims")
    };
    test_slices(ds);

    let DatasetD::D4(ds) = hi.dataset("u_eastward").unwrap() else {
        panic!("wrong dims")
    };
    test_slices(ds);

    let DatasetD::D3(ds) = hi.dataset("Uwind").unwrap() else {
        panic!("wrong dims")
    };
    test_slices(ds);
}
