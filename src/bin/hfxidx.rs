//! Create an index serialized to a flexbuffer.
use std::env;

#[macro_use]
extern crate anyhow;

use flexbuffers::FlexbufferSerializer as ser;
use hidefix::idx::Index;
use serde::ser::Serialize;

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

    print!("Indexing {fin}..");

    let idx = Index::index(fin)?;

    println!("done.");

    print!("Writing index to {fout} (as flxebuffer)..");

    let mut s = ser::new();
    idx.serialize(&mut s)?;
    std::fs::write(fout, s.view())?;

    println!("done.");

    Ok(())
}
