#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::idx::Index;

mod serde_bincode {
    use super::*;

    #[bench]
    fn serialize_coads(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        b.iter(|| bincode::serialize(&i).unwrap())
    }

    #[bench]
    fn deserialize_coads(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let bb = bincode::serialize(&i).unwrap();

        b.iter(|| bincode::deserialize::<Index>(&bb).unwrap())
    }

    #[bench]
    fn serialize_coads_file(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        b.iter(|| {
            let f = std::fs::File::create("/tmp/coads.idx.bc").unwrap();
            let w = std::io::BufWriter::new(f);
            bincode::serialize_into(w, &i).unwrap()
        })
    }

    #[bench]
    fn deserialize_coads_file(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let f = std::fs::File::create("/tmp/coads.idx.bc").unwrap();
        bincode::serialize_into(f, &i).unwrap();

        b.iter(|| {
            let b = std::fs::read("/tmp/coads.idx.bc").unwrap();
            bincode::deserialize::<Index>(&b).unwrap();
        })
    }

    #[ignore]
    #[bench]
    fn serialize_meps_bincode(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();

        b.iter(|| bincode::serialize(&i).unwrap())
    }

    #[ignore]
    #[bench]
    fn deserialize_meps_bincode(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();
        let bb = bincode::serialize(&i).unwrap();

        b.iter(|| bincode::deserialize::<Index>(&bb).unwrap())
    }

    #[ignore]
    #[bench]
    fn serialize_meps_bincode_file(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();

        b.iter(|| {
            let f = std::fs::File::create("/tmp/meps.idx.bc").unwrap();
            let w = std::io::BufWriter::new(f);
            bincode::serialize_into(w, &i).unwrap()
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_meps_bincode_file(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();
        let f = std::fs::File::create("/tmp/meps.idx.bc").unwrap();
        bincode::serialize_into(f, &i).unwrap();

        b.iter(|| {
            let b = std::fs::read("/tmp/meps.idx.bc").unwrap();
            bincode::deserialize::<Index>(&b).unwrap();
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_meps_file_only_read(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();
        let f = std::fs::File::create("/tmp/meps.idx.bc").unwrap();
        bincode::serialize_into(f, &i).unwrap();

        b.iter(|| {
            test::black_box(std::fs::read("/tmp/meps.idx.bc").unwrap());
        })
    }
}

mod serde_flexbuffers {
    use super::*;
    use flexbuffers::FlexbufferSerializer as ser;
    use serde::ser::Serialize;

    #[bench]
    fn serialize_coads(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        b.iter(|| {
            let mut s = ser::new();
            i.serialize(&mut s).unwrap();
        })
    }

    #[bench]
    fn deserialize_coads(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();

        b.iter(|| {
            flexbuffers::from_slice::<Index>(s.view()).unwrap();
        })
    }

    #[bench]
    fn serialize_coads_file(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        b.iter(|| {
            let mut s = ser::new();
            i.serialize(&mut s).unwrap();
            std::fs::write("/tmp/coads.idx.fx", s.view()).unwrap();
        })
    }

    #[bench]
    fn deserialize_coads_file(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/coads.idx.fx", s.view()).unwrap();

        b.iter(|| {
            let b = std::fs::read("/tmp/coads.idx.fx").unwrap();
            flexbuffers::from_slice::<Index>(&b).unwrap();
        })
    }

    #[bench]
    fn deserialize_coads_file_only_read(b: &mut Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/coads.idx.fx", s.view()).unwrap();

        b.iter(|| {
            test::black_box(std::fs::read("/tmp/coads.idx.fx").unwrap());
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_meps_file(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/meps.idx.fx", s.view()).unwrap();

        b.iter(|| {
            let b = std::fs::read("/tmp/meps.idx.fx").unwrap();
            flexbuffers::from_slice::<Index>(&b).unwrap();
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_meps_file_only_read(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/meps.idx.fx", s.view()).unwrap();

        b.iter(|| {
            test::black_box(std::fs::read("/tmp/meps.idx.fx").unwrap());
        })
    }
}
