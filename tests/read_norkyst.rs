#![feature(test)]
#![allow(non_snake_case)]
extern crate test;

use std::path::PathBuf;
use std::sync::Mutex;
use hidefix::prelude::*;

const URL: &'static str = "https://thredds.met.no/thredds/fileServer/fou-hi/norkyst800m/NorKyst-800m_ZDEPTHS_avg.an.2023050800.nc";

fn get_file() ->  PathBuf {
    static NK: Mutex<()> = Mutex::new(());
    let _guard = NK.lock().unwrap();

    let mut p = std::env::temp_dir();
    p.push("hidefix");

    let d = p.clone();

    p.push("norkyst.nc");

    if !p.exists() {
        println!("downloading norkyst file to {p:#?}..");
        std::fs::create_dir_all(&d).unwrap();
        let r = reqwest::blocking::get(URL).unwrap();
        std::fs::write(&p, r.bytes().unwrap()).unwrap();
    }

    p
}

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

#[test]
fn wind() {
    let p = get_file();

    let h = hdf5::File::open(&p).unwrap();
    let Uw = h.dataset("Uwind").unwrap().read_raw::<i32>().unwrap();
    let Vw = h.dataset("Vwind").unwrap().read_raw::<i32>().unwrap();

    let hi = Index::index(&p).unwrap();
    let hUw = hi.reader("Uwind").unwrap().values::<i32>(None, None).unwrap();
    let hVw = hi.reader("Vwind").unwrap().values::<i32>(None, None).unwrap();

    assert_eq!(Uw, hUw);
    assert_eq!(Vw, hVw);
}

#[test]
fn current() {
    let p = get_file();

    let h = hdf5::File::open(&p).unwrap();
    let u = h.dataset("u_eastward").unwrap().read_raw::<f32>().unwrap();
    let v = h.dataset("v_northward").unwrap().read_raw::<f32>().unwrap();

    assert_eq!(u.len(), h.dataset("u_eastward").unwrap().size());

    let hi = Index::index(&p).unwrap();

    assert_eq!(u.len(), hi.dataset("u_eastward").unwrap().size());

    let hu = hi.reader("u_eastward").unwrap().values::<f32>(None, None).unwrap();
    let hv = hi.reader("v_northward").unwrap().values::<f32>(None, None).unwrap();

    assert_eq!(u, hu);
    assert_eq!(v, hv);
}

