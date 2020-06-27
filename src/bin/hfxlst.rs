///! Dump an serialized index to stdout

use std::env;

#[macro_use]
extern crate anyhow;

use hidefix::idx::Index;
use bincode;

fn usage() {
    println!("Usage: hfxdump input.h5.idx");
}

fn main() -> Result<(), anyhow::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        usage();
        return Err(anyhow!("Invalid arguments"));
    }

    let fin = &args[1];

    println!("Loading index from {}..", fin);

    let f = std::fs::File::open("/tmp/meps.idx.bc")?;
    let r = std::io::BufReader::new(f);
    let idx = bincode::deserialize_from::<_, Index>(r)?;

    println!("Datasets (source path: {:?}):\n", idx.path());
    println!("{:4}{:30} shape:", "", "name:");

    for (k, v) in idx.datasets() {
        println!("{:4}{:30} {:?}", "", k, v.shape);
    }

    Ok(())
}

