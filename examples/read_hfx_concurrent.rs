//! This example reads the same dataset from multiple threads at the same time (concurrently).
use hidefix::prelude::*;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let f = &args[1];

    println!("Indexing file: {f}..");
    let i = Arc::new(Index::index(&f)?);

    for var in &args[2..] {
        let i = Arc::clone(&i);

        println!("Reading values from {var}..");

        const ITERATIONS: usize = 100;
        const REPETITIONS: usize = 100;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        pool.scope(move |s| {
            for ii in 0..ITERATIONS {
                let i = Arc::clone(&i);

                s.spawn(move |_| {
                    let mut r = i.reader(var).unwrap();
                    for _ in 0..REPETITIONS {
                        let values = &r.values::<f32>(None, None).unwrap();
                        println!("Iteration: {}, Number of values: {}", ii, values.len());
                        println!(
                            "Iteration: {}, First value: {}",
                            ii,
                            values.first().unwrap()
                        );
                    }
                });
            }
        });
    }

    Ok(())
}
