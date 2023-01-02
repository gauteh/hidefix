use hidefix::prelude::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let f = &args[1];

    println!("Indexing file: {f}..");
    let i = Index::index(&f)?;

    for var in &args[2..] {
        let mut r = i.reader(var).unwrap();

        println!("Reading values from {var}..");
        let values = r.values::<f32>(None, None)?;

        println!("Number of values: {}", values.len());
        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}
