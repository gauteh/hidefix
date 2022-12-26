use hidefix::prelude::*;
use hidefix::reader::uring::*;
use hidefix::idx::DatasetD;
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // Serialize file
    if !Path::new("norkyst.idx").exists() {
        println!("Serializing..");
        let i = Index::index("tests/data/Barents-2.5km_ZDEPTHS_his.an.2022112006.nc")?;
        let idx = bincode::serialize(&i)?;
        fs::write("example.idx", &idx)?;
    }

    println!("Deserializing index..");
    let bb = fs::read("example.idx")?;
    let i: Index = bincode::deserialize(&bb)?;

    for var in ["temperature", "u", "v"] {
        let ds = if let DatasetD::D4(ds) = i.dataset(var).unwrap() {
            ds
        } else {
            panic!()
        };
        // println!("datatype: {:?}", ds.dtype);
        let mut r = UringReader::with_dataset(ds, i.path().unwrap())?;

        println!("Reading values..");
        // let values = tokio_uring::start(async {
        //     r.values_uring::<f32>(None, None).await
        // })?;

        let values = r.values::<f32>(None, None).unwrap();

        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}
