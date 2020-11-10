#![feature(test)]
extern crate test;
use std::sync::Arc;
use test::Bencher;

use hidefix::prelude::*;

// const OVERSUBSCRIBE_THREADS: usize = 50;
const ITERATIONS: usize = 100;
const REPETITIONS: usize = 100;

/// This tests concurrent access to HDF5 files.
///
/// Currently the following scenarios are tested:
///
///     * Sequential reads.
///     * 8 threads trying to read the same dataset.
///
/// In addition, it would be interesting to test:
///
///     * Oversubscribed threads (e.g. 50 threads on 8 core computer)
///     * Using a pool of fds (should not matter on native), but might save some
///       time using `hidefix` since the file is opened once for each iteration now.
///     * Using a pool with oversubscribed workers.
///
///     These additional cases are likely to perform worse on the native reader.
///

mod shuffled_compressed {
    use super::*;

    #[ignore]
    #[bench]
    fn cache_sequential(b: &mut Bencher) {
        let i = Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let mut r = i.reader("d_4_shufzip_chunks").unwrap();

        b.iter(|| {
            for _ in 0..ITERATIONS {
                for _ in 0..REPETITIONS {
                    test::black_box(&r.values::<f32>(None, None).unwrap());
                }
            }
        })
    }

    #[ignore]
    #[bench]
    fn native_sequential(b: &mut Bencher) {
        let h = hdf5::File::open("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let d = h.dataset("d_4_shufzip_chunks").unwrap();

        b.iter(|| {
            for _ in 0..ITERATIONS {
                for _ in 0..REPETITIONS {
                    test::black_box(&d.read_raw::<f32>().unwrap());
                }
            }
        })
    }

    #[ignore]
    #[bench]
    fn cache_concurrent_reads(b: &mut Bencher) {
        let i = Arc::new(Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        b.iter(move || {
            let i = Arc::clone(&i);
            pool.scope(move |s| {
                for _ in 0..ITERATIONS {
                    let i = Arc::clone(&i);

                    s.spawn(move |_| {
                        let mut r = i.reader("d_4_shufzip_chunks").unwrap();
                        for _ in 0..REPETITIONS {
                            test::black_box(&r.values::<f32>(None, None).unwrap());
                        }
                    });
                }
            })
        })
    }

    #[ignore]
    #[bench]
    fn native_concurrent_reads(b: &mut Bencher) {
        let h = hdf5::File::open("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let d = Arc::new(h.dataset("d_4_shufzip_chunks").unwrap());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        b.iter(move || {
            let d = Arc::clone(&d);
            pool.scope(move |s| {
                for _ in 0..ITERATIONS {
                    let d = Arc::clone(&d);

                    s.spawn(move |_| {
                        for _ in 0..REPETITIONS {
                            test::black_box(&d.read_raw::<f32>().unwrap());
                        }
                    });
                }
            })
        })
    }
}
