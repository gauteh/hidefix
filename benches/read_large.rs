#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::idx::Index;

mod meps {
    use super::*;
    use ndarray::s;

    #[ignore]
    #[bench]
    fn idx_small_slice(b: &mut Bencher) {
        let i = Index::index("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let mut r = i.reader("x_wind_ml").unwrap();

        // test against native
        let h = hdf5::File::open("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let d = h.dataset("x_wind_ml").unwrap();
        let hv = d.read_slice::<i32, _, _>(s![0..2, 0..2, 0..1, 0..5]).unwrap().iter().map(|v| *v).collect::<Vec<i32>>();

        assert_eq!(hv, r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[2, 2, 1, 5])).unwrap());

        b.iter(|| test::black_box(r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[2, 2, 1, 5])).unwrap()));
    }

    #[ignore]
    #[bench]
    fn native_small_slice(b: &mut Bencher) {
        let h = hdf5::File::open("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let d = h.dataset("x_wind_ml").unwrap();

        b.iter(|| test::black_box(d.read_slice::<i32, _, _>(s![0..2, 0..2, 0..1, 0..5]).unwrap()))
    }

    #[ignore]
    #[bench]
    fn idx_med_slice(b: &mut Bencher) {
        let i = Index::index("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let mut r = i.reader("x_wind_ml").unwrap();

        // test against native
        let h = hdf5::File::open("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let d = h.dataset("x_wind_ml").unwrap();
        let hv = d.read_slice::<i32, _, _>(s![0..10, 0..10, 0..1, 0..20000]).unwrap().iter().map(|v| *v).collect::<Vec<i32>>();

        assert_eq!(hv, r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[10, 10, 1, 20000])).unwrap());

        b.iter(|| test::black_box(r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[10, 10, 1, 20000])).unwrap()));
    }

    #[ignore]
    #[bench]
    fn native_med_slice(b: &mut Bencher) {
        let h = hdf5::File::open("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let d = h.dataset("x_wind_ml").unwrap();

        b.iter(|| test::black_box(d.read_slice::<i32, _, _>(s![0..10, 0..10, 0..1, 0..20000]).unwrap()))
    }

    #[ignore]
    #[bench]
    fn idx_big_slice(b: &mut Bencher) {
        let i = Index::index("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let mut r = i.reader("x_wind_ml").unwrap();

        // test against native
        let h = hdf5::File::open("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let d = h.dataset("x_wind_ml").unwrap();
        let hv = d.read_slice::<i32, _, _>(s![0..65, 0..65, 0..1, 0..20000]).unwrap().iter().map(|v| *v).collect::<Vec<i32>>();

        assert_eq!(hv, r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[65, 65, 1, 20000])).unwrap());

        b.iter(|| test::black_box(r.values::<i32>(Some(&[0, 0, 0, 0]), Some(&[65, 65, 1, 20000])).unwrap()));
    }

    #[ignore]
    #[bench]
    fn native_big_slice(b: &mut Bencher) {
        let h = hdf5::File::open("../data/meps_det_vc_2_5km_latest.nc").unwrap();
        let d = h.dataset("x_wind_ml").unwrap();

        b.iter(|| test::black_box(d.read_slice::<i32, _, _>(s![0..65, 0..65, 0..1, 0..20000]).unwrap()))
    }
}
