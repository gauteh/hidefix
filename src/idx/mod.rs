mod chunk;
mod dataset;
mod index;
pub mod serde;

pub use chunk::{Chunk, ULE};
pub use dataset::{Dataset, DatasetD, DatasetExt, Datatype};
pub use index::Index;

/// Convenience trait for returning an index for an existing HDF5 file or dataset opened with
/// the standard rust HDF5 library.
///
/// ```
/// use hidefix::prelude::*;
///
/// let hf = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
/// let idx = hf.index().unwrap();
///
/// let mut r = idx.reader("SST").unwrap();
/// let values = r.values::<f32>(None, None).unwrap();
///
/// println!("SST: {:?}", values);
/// ```
pub trait IntoIndex {
    type Indexed;

    fn index(&self) -> anyhow::Result<Self::Indexed>;
}

impl IntoIndex for hdf5::File {
    type Indexed = Index<'static>;

    fn index(&self) -> anyhow::Result<Self::Indexed> {
        self.try_into()
    }
}

impl IntoIndex for hdf5::Dataset {
    type Indexed = DatasetD<'static>;

    fn index(&self) -> anyhow::Result<Self::Indexed> {
        // TODO: use H5get_name to get file name (not really any spot to store the name)
        DatasetD::index(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn into_idx_file() {
        let hf = hdf5::File::open("tests/data/dmrpp/chunked_twoD.h5").unwrap();
        let i = hf.index().unwrap();
        println!("index: {:#?}", i);
    }

    #[test]
    fn into_idx_dataset() {
        let hf = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
        let ds = hf.dataset("SST").unwrap();
        let i = ds.index().unwrap();
        println!("dataset index: {:#?}", i);
    }
}
