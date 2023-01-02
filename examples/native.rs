fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let f = &args[1];
    let h = hdf5::File::open(&f).unwrap();

    for var in &args[2..] {
        let d = h.dataset(var)?;

        println!("Reading values from {var}..");
        let values = d.read_raw::<f32>()?;

        println!("Number of values: {}", values.len());

        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}
