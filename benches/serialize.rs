use divan::Bencher;

use hidefix::idx::Index;

const FILE: Option<&'static str> = option_env!("HIDEFIX_LARGE_FILE");
// const VAR: &'static str = env!("HIDEFIX_LARGE_VAR");

mod serde_bincode {
    use super::*;

    #[divan::bench]
    fn serialize_coads(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        b.bench_local(|| bincode::serialize(&i).unwrap())
    }

    #[divan::bench]
    fn deserialize_coads(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let bb = bincode::serialize(&i).unwrap();

        b.bench_local(|| bincode::deserialize::<Index>(&bb).unwrap())
    }

    #[divan::bench]
    fn serialize_coads_file(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        b.bench_local(|| {
            let f = std::fs::File::create("/tmp/coads.idx.bc").unwrap();
            let w = std::io::BufWriter::new(f);
            bincode::serialize_into(w, &i).unwrap()
        })
    }

    #[divan::bench]
    fn deserialize_coads_file(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let f = std::fs::File::create("/tmp/coads.idx.bc").unwrap();
        bincode::serialize_into(f, &i).unwrap();

        b.bench_local(|| {
            let b = std::fs::read("/tmp/coads.idx.bc").unwrap();
            bincode::deserialize::<Index>(&b).unwrap();
        })
    }

    #[ignore]
    #[divan::bench]
    fn serialize_large_bincode(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();

        b.bench_local(|| bincode::serialize(&i).unwrap())
    }

    #[ignore]
    #[divan::bench]
    fn deserialize_large_bincode(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let bb = bincode::serialize(&i).unwrap();

        b.bench_local(|| bincode::deserialize::<Index>(&bb).unwrap())
    }

    #[ignore]
    #[divan::bench]
    fn serialize_large_bincode_file(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();

        b.bench_local(|| {
            let f = std::fs::File::create("/tmp/large.idx.bc").unwrap();
            let w = std::io::BufWriter::new(f);
            bincode::serialize_into(w, &i).unwrap()
        })
    }

    #[ignore]
    #[divan::bench]
    fn deserialize_large_bincode_file(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let f = std::fs::File::create("/tmp/large.idx.bc").unwrap();
        bincode::serialize_into(f, &i).unwrap();

        b.bench_local(|| {
            let b = std::fs::read("/tmp/large.idx.bc").unwrap();
            bincode::deserialize::<Index>(&b).unwrap();
        })
    }

    #[ignore]
    #[divan::bench]
    fn deserialize_large_bincode_db_sled(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();

        let bts = bincode::serialize(&i).unwrap();

        let db = sled::Config::default()
            .temporary(true)
            .print_profile_on_drop(true)
            .open()
            .unwrap();

        db.insert("large", bts).unwrap();

        b.bench_local(|| {
            let bts = db.get("large").unwrap().unwrap();
            divan::black_box(bincode::deserialize::<Index>(&bts).unwrap());
        })
    }

    #[ignore]
    #[divan::bench]
    fn deserialize_large_bincode_db_sled_only_read(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();

        let bts = bincode::serialize(&i).unwrap();

        let db = sled::Config::default()
            .temporary(true)
            .print_profile_on_drop(true)
            .open()
            .unwrap();

        db.insert("large", bts).unwrap();

        b.bench_local(|| {
            divan::black_box(db.get("large").unwrap().unwrap());
        })
    }

    #[ignore]
    #[divan::bench]
    fn deserialize_large_file_only_read(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let f = std::fs::File::create("/tmp/large.idx.bc").unwrap();
        bincode::serialize_into(f, &i).unwrap();

        b.bench_local(|| {
            divan::black_box(std::fs::read("/tmp/large.idx.bc").unwrap());
        })
    }
}

mod serde_flexbuffers {
    use super::*;
    use flexbuffers::FlexbufferSerializer as ser;
    use serde::ser::Serialize;

    #[divan::bench]
    fn serialize_coads(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        b.bench_local(|| {
            let mut s = ser::new();
            i.serialize(&mut s).unwrap();
        })
    }

    #[divan::bench]
    fn deserialize_coads(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();

        b.bench_local(|| {
            flexbuffers::from_slice::<Index>(s.view()).unwrap();
        })
    }

    #[divan::bench]
    fn serialize_coads_file(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        b.bench_local(|| {
            let mut s = ser::new();
            i.serialize(&mut s).unwrap();
            std::fs::write("/tmp/coads.idx.fx", s.view()).unwrap();
        })
    }

    #[divan::bench]
    fn deserialize_coads_file(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/coads.idx.fx", s.view()).unwrap();

        b.bench_local(|| {
            let b = std::fs::read("/tmp/coads.idx.fx").unwrap();
            flexbuffers::from_slice::<Index>(&b).unwrap();
        })
    }

    #[divan::bench]
    fn deserialize_coads_file_only_read(b: Bencher) {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/coads.idx.fx", s.view()).unwrap();

        b.bench_local(|| {
            divan::black_box(std::fs::read("/tmp/coads.idx.fx").unwrap());
        })
    }

    #[ignore]
    #[divan::bench]
    fn deserialize_large_file(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/large.idx.fx", s.view()).unwrap();

        b.bench_local(|| {
            let b = std::fs::read("/tmp/large.idx.fx").unwrap();
            flexbuffers::from_slice::<Index>(&b).unwrap();
        })
    }

    #[ignore]
    #[divan::bench]
    fn deserialize_large_file_only_read(b: Bencher) {
        let i = Index::index(FILE.unwrap()).unwrap();
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        std::fs::write("/tmp/large.idx.fx", s.view()).unwrap();

        b.bench_local(|| {
            divan::black_box(std::fs::read("/tmp/large.idx.fx").unwrap());
        })
    }
}

fn main() {
    divan::main();
}
