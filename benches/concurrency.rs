use divan::Bencher;
use std::sync::Arc;

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
    #[divan::bench]
    fn cache_sequential(b: Bencher) {
        let i = Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let mut r = i.reader("d_4_shufzip_chunks").unwrap();

        b.bench_local(|| {
            for _ in 0..ITERATIONS {
                for _ in 0..REPETITIONS {
                    divan::black_box(&r.values::<f32, _>(..).unwrap());
                }
            }
        })
    }

    #[ignore]
    #[divan::bench]
    fn direct_sequential_parallel(b: Bencher) {
        let i = Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let ds = i.dataset("d_4_shufzip_chunks").unwrap();
        let r = ds
            .as_par_reader(&"tests/data/dmrpp/chunked_shufzip_twoD.h5")
            .unwrap();

        b.bench_local(|| {
            for _ in 0..ITERATIONS {
                for _ in 0..REPETITIONS {
                    divan::black_box(&r.values_par::<f32, _>(..).unwrap());
                }
            }
        })
    }

    #[ignore]
    #[divan::bench]
    fn native_sequential(b: Bencher) {
        let h = hdf5::File::open("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let d = h.dataset("d_4_shufzip_chunks").unwrap();

        b.bench_local(|| {
            for _ in 0..ITERATIONS {
                for _ in 0..REPETITIONS {
                    divan::black_box(&d.read_raw::<f32>().unwrap());
                }
            }
        })
    }

    #[ignore]
    #[divan::bench]
    fn cache_concurrent_reads(b: Bencher) {
        let i = Arc::new(Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        b.bench_local(move || {
            let i = Arc::clone(&i);
            pool.scope(move |s| {
                for _ in 0..ITERATIONS {
                    let i = Arc::clone(&i);

                    s.spawn(move |_| {
                        let mut r = i.reader("d_4_shufzip_chunks").unwrap();
                        for _ in 0..REPETITIONS {
                            divan::black_box(&r.values::<f32, _>(..).unwrap());
                        }
                    });
                }
            })
        })
    }

    #[ignore]
    #[divan::bench]
    fn native_concurrent_reads(b: Bencher) {
        let h = hdf5::File::open("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let d = Arc::new(h.dataset("d_4_shufzip_chunks").unwrap());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        b.bench_local(move || {
            let d = Arc::clone(&d);
            pool.scope(move |s| {
                for _ in 0..ITERATIONS {
                    let d = Arc::clone(&d);

                    s.spawn(move |_| {
                        for _ in 0..REPETITIONS {
                            divan::black_box(&d.read_raw::<f32>().unwrap());
                        }
                    });
                }
            })
        })
    }
}

fn main() {
    divan::main();
}
