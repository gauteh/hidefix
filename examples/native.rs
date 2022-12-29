fn main() -> anyhow::Result<()> {
    let h = hdf5::File::open("tests/data/Barents-2.5km_ZDEPTHS_his.an.2022112006.nc").unwrap();

    for var in ["temperature", "u", "v"] {
        let d = h.dataset(var)?;

        println!("Reading values..");
        let values = d.read_raw::<f32>()?;

        println!("Number of values: {}", values.len());

        println!("First value: {}", values.first().unwrap());
    }

    Ok(())
}

