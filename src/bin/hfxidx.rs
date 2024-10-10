//! Create an index serialized to a file on disk.
use clap::Parser;
use hidefix::idx::Index;

#[derive(Parser, Debug)]
struct Args {
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    #[arg(default_value = "any", long)]
    out_type: String,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let fin = &args.input;
    let fout = &args.output;

    print!("Indexing {}..", fin.to_string_lossy());

    let idx = Index::index(fin)?;

    println!("done.");

    println!(
        "Writing index to {} (as {})..",
        fout.to_string_lossy(),
        args.out_type,
    );

    #[allow(unreachable_patterns)] // To let "any" work
    match args.out_type.as_str() {
        #[cfg(feature = "flexbuffers")]
        "any" | "flexbuffers" => {
            use serde::ser::Serialize;
            let mut s = flexbuffers::FlexbufferSerializer::new();
            idx.serialize(&mut s)?;
            std::fs::write(fout, s.view())?;
        }
        #[cfg(feature = "bincode")]
        "any" | "bincode" => {
            let s = bincode::serialize(&idx)?;
            std::fs::write(fout, s)?;
        }
        "any" => anyhow::bail!("No serializer compiled in"),
        _ => anyhow::bail!("Unknown serialization type {}", args.out_type),
    }

    println!("done.");

    Ok(())
}
