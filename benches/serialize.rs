#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::idx::Index;

const FILE: Option<&'static str> = option_env!("HIDEFIX_LARGE_FILE");
// const VAR: &'static str = env!("HIDEFIX_LARGE_VAR");

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
    fn serialize_large_bincode(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();

        b.iter(|| bincode::serialize(&i).unwrap())
    }

    #[ignore]
    #[bench]
    fn deserialize_large_bincode(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let bb = bincode::serialize(&i).unwrap();

        b.iter(|| bincode::deserialize::<Index>(&bb).unwrap())
    }

    #[ignore]
    #[bench]
    fn serialize_large_bincode_file(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();

        b.iter(|| {
            let f = std::fs::File::create("/tmp/large.idx.bc").unwrap();
            let w = std::io::BufWriter::new(f);
            bincode::serialize_into(w, &i).unwrap()
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_large_bincode_file(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let f = std::fs::File::create("/tmp/large.idx.bc").unwrap();
        bincode::serialize_into(f, &i).unwrap();

        b.iter(|| {
            let b = std::fs::read("/tmp/large.idx.bc").unwrap();
            bincode::deserialize::<Index>(&b).unwrap();
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_large_bincode_db_sled(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();

        let bts = bincode::serialize(&i).unwrap();

        let db = sled::Config::default()
            .temporary(true)
            .print_profile_on_drop(true)
            .open()
            .unwrap();

        db.insert("large", bts).unwrap();

        b.iter(|| {
            let bts = db.get("large").unwrap().unwrap();
            test::black_box(bincode::deserialize::<Index>(&bts).unwrap());
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_large_bincode_db_sled_only_read(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();

        let bts = bincode::serialize(&i).unwrap();

        let db = sled::Config::default()
            .temporary(true)
            .print_profile_on_drop(true)
            .open()
            .unwrap();

        db.insert("large", bts).unwrap();

        b.iter(|| {
            test::black_box(db.get("large").unwrap().unwrap());
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_large_file_only_read(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let f = std::fs::File::create("/tmp/large.idx.bc").unwrap();
        bincode::serialize_into(f, &i).unwrap();

        b.iter(|| {
            test::black_box(std::fs::read("/tmp/large.idx.bc").unwrap());
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
    fn deserialize_large_file(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/large.idx.fx", s.view()).unwrap();

        b.iter(|| {
            let b = std::fs::read("/tmp/large.idx.fx").unwrap();
            flexbuffers::from_slice::<Index>(&b).unwrap();
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_large_file_only_read(b: &mut Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/large.idx.fx", s.view()).unwrap();

        b.iter(|| {
            test::black_box(std::fs::read("/tmp/large.idx.fx").unwrap());
        })
    }
}
