use hidefix::prelude::*;
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // Serialize file
    if !Path::new("example.idx").exists() {
        println!("Serializing..");
        let i = Index::index("tests/data/Barents-2.5km_ZDEPTHS_his.an.2022112006.nc")?;
        let idx = bincode::serialize(&i)?;
        fs::write("example.idx", &idx)?;
    }

    println!("Deserializing index..");
    let bb = fs::read("example.idx")?;
    let i: Index = bincode::deserialize(&bb)?;
    for var in ["temperature", "u", "v"] {
        let mut r = i.reader(var).unwrap();

        println!("Reading values..");
        let values = r.values::<f32>(None, None)?;

        println!("Number of values: {}", values.len());
        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}

