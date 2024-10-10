//! List a summary of a flexbuffer serialized index to stdout.
use clap::Parser;
use hidefix::idx::{DatasetExt, Index};

#[derive(Parser, Debug)]
struct Args {
    input: std::path::PathBuf,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    println!("Loading index from {}..", args.input.to_string_lossy());

    let b = std::fs::read(&args.input)?;
    let idx = read_index(&b)?;

    let path = idx.path();
    if let Some(path) = path {
        println!("Datasets (source path: {path:?})");
    } else {
        println!("Datasets");
    }
    for (name, dataset) in idx.datasets() {
        let shape = dataset.shape();
        println!("{name:32} {shape:?}");
    }
    // println!("{idx:?}");

    Ok(())
}

fn read_index(bytes: &[u8]) -> Result<Index, anyhow::Error> {
    let mut errs: Vec<anyhow::Error> = vec![];
    #[cfg(feature = "flexbuffers")]
    {
        match read_flexbuffer(bytes) {
            Ok(idx) => return Ok(idx),
            Err(e) => errs.push(e),
        }
    }
    #[cfg(feature = "bincode")]
    {
        match read_bincode(bytes) {
            Ok(idx) => return Ok(idx),
            Err(e) => errs.push(e),
        }
    }
    anyhow::bail!("Parsing failed (incompatible, or not configured): {errs:?}")
}

#[cfg(feature = "flexbuffers")]
fn read_flexbuffer(bytes: &[u8]) -> Result<Index, anyhow::Error> {
    let idx = flexbuffers::from_slice(bytes)?;
    Ok(idx)
}

#[cfg(feature = "bincode")]
fn read_bincode(bytes: &[u8]) -> Result<Index, anyhow::Error> {
    let idx = bincode::deserialize(bytes)?;
    Ok(idx)
}
