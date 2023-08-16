#![feature(test)]
extern crate test;
use test::Bencher;

use std::path::PathBuf;
use std::sync::Mutex;

use hidefix::prelude::*;

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
#[bench]
fn idx_big_slice(b: &mut Bencher) {
    let p = get_file();
    let i = Index::index(&p).unwrap();
    let mut u = i.reader("u_eastward").unwrap();

    b.iter(|| test::black_box(u.values::<f32>(None, None).unwrap()));
}

#[ignore]
#[bench]
fn native_big_slice(b: &mut Bencher) {
    let p = get_file();
    let h = hdf5::File::open(&p).unwrap();
    let d = h.dataset("u_eastward").unwrap();

    b.iter(|| test::black_box(d.read_raw::<f32>().unwrap()))
}
