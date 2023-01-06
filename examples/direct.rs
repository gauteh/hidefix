use hidefix::prelude::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let f = &args[1];

    println!("Indexing file: {f}..");
    let i = Index::index(&f)?;

    for var in &args[2..] {
        let ds = i.dataset(var).unwrap();
        println!("Datatype: {:?}", ds.dtype());
        let r = ds.as_par_reader(f)?;

        println!("Reading values from {var}..");
        // let values = r.values_par::<f32>(None, None).unwrap();
        let values = r.values_dyn_par::<f32>(None, None).unwrap();

        println!("Number of values: {}", values.len());
        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}
