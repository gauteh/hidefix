#![cfg(feature = "s3")]
//! Integration tests for the S3 reader against a local Minio instance.
//!
//! Start Minio with:
//!
//! ```sh
//! docker compose -f tests/docker-compose.minio.yml up -d
//! ```
//!
//! and run the tests with:
//!
//! ```sh
//! HIDEFIX_S3_ENDPOINT=http://localhost:9000 cargo test --features s3 --test read_s3
//! ```
//!
//! The tests are skipped when `HIDEFIX_S3_ENDPOINT` is not set.

use std::sync::{Mutex, OnceLock};

use hidefix::idx::DatasetD;
use hidefix::prelude::*;
use hidefix::reader::s3::S3Reader;
use s3::creds::Credentials;
use s3::{Bucket, BucketConfiguration, Region};

const BUCKET: &str = "hidefix-test";

const FILES: &[&str] = &[
    "tests/data/coads_climatology.nc4",
    "tests/data/dmrpp/chunked_oneD.h5",
    "tests/data/dmrpp/chunked_gzipped_twoD.h5",
    "tests/data/dmrpp/chunked_shuffled_twoD.h5",
];

fn rt() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| tokio::runtime::Runtime::new().unwrap())
}

fn region() -> Option<Region> {
    let endpoint = std::env::var("HIDEFIX_S3_ENDPOINT").ok()?;
    Some(Region::Custom {
        region: "us-east-1".into(),
        endpoint,
    })
}

fn credentials() -> Credentials {
    let access = std::env::var("HIDEFIX_S3_ACCESS_KEY").unwrap_or_else(|_| "minioadmin".into());
    let secret = std::env::var("HIDEFIX_S3_SECRET_KEY").unwrap_or_else(|_| "minioadmin".into());
    Credentials::new(Some(&access), Some(&secret), None, None, None).unwrap()
}

/// Bucket with the test files uploaded, or `None` when no endpoint is configured.
fn bucket() -> Option<Box<Bucket>> {
    let Some(region) = region() else {
        eprintln!("HIDEFIX_S3_ENDPOINT not set, skipping");
        return None;
    };

    let bucket = Bucket::new(BUCKET, region.clone(), credentials())
        .unwrap()
        .with_path_style();

    static SETUP: Mutex<bool> = Mutex::new(false);
    let mut done = SETUP.lock().unwrap();
    if !*done {
        // On a dedicated thread so that `block_on` also works when the calling test runs
        // inside a tokio runtime.
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    rt().block_on(async {
                        if !bucket.exists().await.unwrap() {
                            Bucket::create_with_path_style(
                                BUCKET,
                                region,
                                credentials(),
                                BucketConfiguration::default(),
                            )
                            .await
                            .unwrap();
                        }

                        for f in FILES {
                            bucket
                                .put_object(f, &std::fs::read(f).unwrap())
                                .await
                                .unwrap();
                        }
                    })
                })
                .join()
                .unwrap();
        });
        *done = true;
    }

    Some(bucket)
}

#[test]
fn read_coads_sst() {
    let Some(bucket) = bucket() else { return };

    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let DatasetD::D3(ds) = i.dataset("SST").unwrap() else {
        panic!()
    };
    let mut r = S3Reader::with_dataset(ds, bucket, "tests/data/coads_climatology.nc4").unwrap();

    let vs = r.values::<f32, _>(..).unwrap();

    let h = hdf5::File::open(i.path().unwrap()).unwrap();
    let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

    assert_eq!(vs, hvs);
}

#[test]
fn read_coads_sst_slice() {
    let Some(bucket) = bucket() else { return };

    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let DatasetD::D3(ds) = i.dataset("SST").unwrap() else {
        panic!()
    };
    let mut r = S3Reader::with_dataset(ds, bucket, "tests/data/coads_climatology.nc4").unwrap();

    let extents = [3..7, 10..80, 0..90];
    let vs = r.values::<f32, _>(extents.clone()).unwrap();

    let mut local = i.reader("SST").unwrap();
    let lvs = local.values::<f32, _>(extents).unwrap();

    assert_eq!(vs, lvs);
}

#[test]
fn read_chunked_1d() {
    let Some(bucket) = bucket() else { return };

    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let DatasetD::D1(ds) = i.dataset("d_4_chunks").unwrap() else {
        panic!()
    };
    let mut r = S3Reader::with_dataset(ds, bucket, "tests/data/dmrpp/chunked_oneD.h5").unwrap();

    let vs = r.values::<f32, _>(..).unwrap();

    let h = hdf5::File::open(i.path().unwrap()).unwrap();
    let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

    assert_eq!(vs, hvs);
}

#[test]
fn read_chunked_gzipped_2d() {
    let Some(bucket) = bucket() else { return };

    let i = Index::index("tests/data/dmrpp/chunked_gzipped_twoD.h5").unwrap();
    let DatasetD::D2(ds) = i.dataset("d_4_gzipped_chunks").unwrap() else {
        panic!()
    };
    let mut r =
        S3Reader::with_dataset(ds, bucket, "tests/data/dmrpp/chunked_gzipped_twoD.h5").unwrap();

    let vs = r.values::<f32, _>(..).unwrap();

    let h = hdf5::File::open(i.path().unwrap()).unwrap();
    let hvs = h
        .dataset("d_4_gzipped_chunks")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();

    assert_eq!(vs, hvs);
}

#[test]
fn read_chunked_shuffled_2d() {
    let Some(bucket) = bucket() else { return };

    let i = Index::index("tests/data/dmrpp/chunked_shuffled_twoD.h5").unwrap();
    let DatasetD::D2(ds) = i.dataset("d_4_shuffled_chunks").unwrap() else {
        panic!()
    };
    let mut r =
        S3Reader::with_dataset(ds, bucket, "tests/data/dmrpp/chunked_shuffled_twoD.h5").unwrap();

    let vs = r.values::<f32, _>(..).unwrap();

    let h = hdf5::File::open(i.path().unwrap()).unwrap();
    let hvs = h
        .dataset("d_4_shuffled_chunks")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();

    assert_eq!(vs, hvs);
}

/// The sync `Reader` interface should also work from within a current-thread runtime
/// (`#[tokio::test]` default), where `block_in_place` is not available.
#[tokio::test]
async fn read_chunked_1d_current_thread_runtime() {
    let Some(bucket) = bucket() else { return };

    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let DatasetD::D1(ds) = i.dataset("d_4_chunks").unwrap() else {
        panic!()
    };
    let mut r = S3Reader::with_dataset(ds, bucket, "tests/data/dmrpp/chunked_oneD.h5").unwrap();

    let vs = r.values::<f32, _>(..).unwrap();

    let h = hdf5::File::open(i.path().unwrap()).unwrap();
    let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

    assert_eq!(vs, hvs);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_coads_sst_async() {
    let Some(bucket) = bucket() else { return };

    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let DatasetD::D3(ds) = i.dataset("SST").unwrap() else {
        panic!()
    };
    let r = S3Reader::with_dataset(ds, bucket, "tests/data/coads_climatology.nc4").unwrap();

    let h = hdf5::File::open(i.path().unwrap()).unwrap();
    let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

    let mut dst = vec![0u8; hvs.len() * 4];
    let sz = r.read_to_async(&(..).into(), &mut dst).await.unwrap();
    assert_eq!(sz, dst.len());

    // The sync `Reader` interface should also work from within a multi-threaded runtime.
    let mut r = r;
    let vs = r.values::<f32, _>(..).unwrap();
    assert_eq!(vs, hvs);
}
