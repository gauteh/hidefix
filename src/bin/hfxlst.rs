//! List a summary of a flexbuffer serialized index to stdout.
use std::env;

#[macro_use]
extern crate anyhow;

fn usage() {
    println!("Usage: hfxlst input.h5.idx");
}

fn main() -> Result<(), anyhow::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        usage();
        return Err(anyhow!("Invalid arguments"));
    }

    let fin = &args[1];

    println!("Loading index from {fin}..");

    let b = std::fs::read(fin)?;
    let idx = flexbuffers::Reader::get_root(&*b)?.as_map();

    println!("Datasets (source path: {:?}):\n", idx.idx("path").as_str());
    println!("{:4}{:30} shape:", "", "name:");

    let datasets = idx.idx("datasets").as_map();

    datasets.iter_keys().for_each(|k| {
        let shape: Vec<u64> = datasets
            .idx(k)
            .as_map()
            .idx("shape")
            .as_vector()
            .iter()
            .map(|r| r.as_u64())
            .collect();
        println!("{:4}{:30} {:?}", "", k, shape);
    });

    Ok(())
}
