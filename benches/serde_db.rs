#![feature(test)]
extern crate test;
use test::Bencher;

use hidefix::idx::Index;

mod serde_sled {
    use super::*;

    #[ignore]
    #[bench]
    fn deserialize_meps_bincode_db_sled(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();

        let bts = bincode::serialize(&i).unwrap();

        let db = sled::Config::default()
            .temporary(true)
            .print_profile_on_drop(true)
            .open()
            .unwrap();

        db.insert("meps", bts).unwrap();

        b.iter(|| {
            let bts = db.get("meps").unwrap().unwrap();
            test::black_box(bincode::deserialize::<Index>(&bts).unwrap());
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_meps_bincode_db_sled_only_read(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();

        let bts = bincode::serialize(&i).unwrap();

        let db = sled::Config::default()
            .temporary(true)
            .print_profile_on_drop(true)
            .open()
            .unwrap();

        db.insert("meps", bts).unwrap();

        b.iter(|| {
            test::black_box(db.get("meps").unwrap().unwrap());
        })
    }
}

mod serde_sqlite {
    use super::*;
}
