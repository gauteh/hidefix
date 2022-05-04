#![feature(test)]
extern crate test;
use tempfile::{NamedTempFile, TempDir};
use test::Bencher;

use hidefix::idx::Index;

mod serde_db_sled {
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
    fn deserialize_meps_bincode_only_read(b: &mut Bencher) {
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

mod serde_db_sqlite {
    use super::*;
    use sqlx::SqlitePool;

    #[ignore]
    #[bench]
    fn deserialize_meps_bincode_only_read(b: &mut Bencher) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();

        let bts = bincode::serialize(&i).unwrap();

        println!("serialized size: {}", bts.len());

        let db = NamedTempFile::new().unwrap();

        let pool = rt.block_on(async {
            let pool = SqlitePool::connect(&format!("sqlite:{}", db.path().to_str().unwrap()))
                .await
                .unwrap();
            let mut c = pool.acquire().await.unwrap();

            sqlx::query("CREATE TABLE hdf5 (dataset TEXT, idx BLOB)")
                .execute(&mut c)
                .await
                .unwrap();
            sqlx::query("CREATE INDEX dataset_idx ON hdf5 (dataset)")
                .execute(&mut c)
                .await
                .unwrap();

            // Insert index into db
            sqlx::query("INSERT INTO hdf5 (dataset, idx) VALUES (?1, ?2)")
                .bind("meps")
                .bind(&bts)
                .execute(&mut c)
                .await
                .unwrap();

            pool
        });

        b.iter(|| {
            let (_nbts,): (Vec<u8>,) = test::black_box(
                rt.block_on(
                    sqlx::query_as("SELECT idx FROM hdf5 WHERE dataset = ?1")
                        .bind("meps")
                        .fetch_one(&pool),
                )
                .unwrap(),
            );
            // assert_eq!(&bts, &nbts);

            // let bts = rt.block_on(async { test::black_box(Vec::<u8>::with_capacity(8_000_000)) });
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_meps_bincode(b: &mut Bencher) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();

        let bts = bincode::serialize(&i).unwrap();

        println!("serialized size: {}", bts.len());

        let db = NamedTempFile::new().unwrap();

        let pool = rt.block_on(async {
            let pool = SqlitePool::connect(&format!("sqlite:{}", db.path().to_str().unwrap()))
                .await
                .unwrap();
            let mut c = pool.acquire().await.unwrap();

            sqlx::query("CREATE TABLE hdf5 (dataset TEXT, idx BLOB)")
                .execute(&mut c)
                .await
                .unwrap();
            sqlx::query("CREATE INDEX dataset_idx ON hdf5 (dataset)")
                .execute(&mut c)
                .await
                .unwrap();

            // Insert index into db
            sqlx::query("INSERT INTO hdf5 (dataset, idx) VALUES (?1, ?2)")
                .bind("meps")
                .bind(&bts)
                .execute(&mut c)
                .await
                .unwrap();

            pool
        });

        b.iter(|| {
            let (nbts,): (Vec<u8>,) = rt
                .block_on(
                    sqlx::query_as("SELECT idx FROM hdf5 WHERE dataset = ?1")
                        .bind("meps")
                        .fetch_one(&pool),
                )
                .unwrap();
            // assert_eq!(&bts, &nbts);
            test::black_box(bincode::deserialize::<Index>(&nbts).unwrap());
        })
    }
}

mod serde_db_heed {
    use super::*;
    use heed::types::*;
    use heed::{Database, EnvOpenOptions};

    #[ignore]
    #[bench]
    fn deserialize_meps_bincode(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();

        let bts = bincode::serialize(&i).unwrap();

        let dbp = TempDir::new().unwrap();
        let env = EnvOpenOptions::new()
            .map_size(10 * 1024 * 1024)
            .open(dbp)
            .unwrap();

        let db: Database<Str, ByteSlice> = env.create_database(None).unwrap();

        let mut wtxn = env.write_txn().unwrap();
        db.put(&mut wtxn, "meps", &bts).unwrap();
        wtxn.commit().unwrap();

        b.iter(|| {
            let rtxn = env.read_txn().unwrap();
            let nbts = db.get(&rtxn, "meps").unwrap().unwrap();
            test::black_box(bincode::deserialize::<Index>(&nbts).unwrap());
        })
    }

    #[ignore]
    #[bench]
    fn deserialize_meps_bincode_only_read(b: &mut Bencher) {
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();

        let bts = bincode::serialize(&i).unwrap();

        let dbp = TempDir::new().unwrap();
        let env = EnvOpenOptions::new()
            .map_size(1000 * 1024 * 1024)
            .open(dbp)
            .unwrap();

        let db: Database<Str, ByteSlice> = env.create_database(None).unwrap();

        let mut wtxn = env.write_txn().unwrap();
        db.put(&mut wtxn, "meps", &bts).unwrap();
        wtxn.commit().unwrap();

        b.iter(|| {
            let rtxn = env.read_txn().unwrap();
            test::black_box(db.get(&rtxn, "meps").unwrap().unwrap());
        })
    }
}
