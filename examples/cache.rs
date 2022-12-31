use hidefix::prelude::*;
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("Indexing file..");
    let i = Index::index("tests/data/Barents-2.5km_ZDEPTHS_his.an.2022112006.nc")?;

    for var in ["temperature", "u", "v"] {
        let mut r = i.reader(var).unwrap();

        println!("Reading values..");
        let values = r.values::<f32>(None, None)?;

        println!("Number of values: {}", values.len());
        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}

