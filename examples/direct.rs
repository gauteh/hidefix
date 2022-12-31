use hidefix::prelude::*;
use hidefix::reader::direct::*;
use hidefix::idx::DatasetD;
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("Indexing file..");
    let i = Index::index("tests/data/Barents-2.5km_ZDEPTHS_his.an.2022112006.nc")?;

    for var in ["temperature", "u", "v"] {
        let ds = if let DatasetD::D4(ds) = i.dataset(var).unwrap() {
            ds
        } else {
            panic!()
        };
        // println!("datatype: {:?}", ds.dtype);
        let mut r = Direct::with_dataset(ds, i.path().unwrap())?;

        println!("Reading values..");
        let values = r.values::<f32>(None, None).unwrap();

        println!("Number of values: {}", values.len());
        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}
