///! Create an index serialized to bincode

use std::env;

#[macro_use]
extern crate anyhow;

use hidefix::idx::Index;
use bincode;

fn usage() {
    println!("Usage: hfxidx input.h5 output.h5.idx");
}

fn main() -> Result<(), anyhow::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        usage();
        return Err(anyhow!("Invalid arguments"));
    }

    let fin = &args[1];
    let fout = &args[2];

    println!("Indexing {}..", fin);

    let idx = Index::index(fin)?;

    println!("Writing index to {} (as bincode)..", fout);
    let f = std::fs::File::create(fout)?;
    let w = std::io::BufWriter::new(f);
    bincode::serialize_into(w, &idx)?;

    println!("Done.");

    Ok(())
}

