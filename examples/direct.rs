use hidefix::prelude::*;
use hidefix::reader::direct::*;
use hidefix::idx::DatasetD;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let f = &args[1];

    println!("Indexing file: {f}..");
    let i = Index::index(&f)?;

    for var in &args[2..] {
        let ds = if let DatasetD::D4(ds) = i.dataset(var).unwrap() {
            ds
        } else {
            panic!()
        };
        println!("Datatype: {:?}", ds.dtype);
        let r = Direct::with_dataset(ds, i.path().unwrap())?;

        println!("Reading values from {var}..");
        let values = r.values_par::<f32>(None, None).unwrap();

        println!("Number of values: {}", values.len());
        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}
