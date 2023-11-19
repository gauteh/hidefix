//! # HIDEFIX
//!
//! A fast and concurrent reader for HDF5 and NetCDF (v4) files.
//!
//! This library allows a HDF5 file to be read in a multi-threaded or concurrent (async) way. The
//! chunks of a dataset need to be indexed in advance. Fast in newer versions of HDF5 (see below).
//! The index can be efficiently deserialized with zero-copy through [serde](https://serde.rs/).
//!
//! This allows multiple [datasets](idx::Dataset) (variables) to be read at the same time, or even different
//! domains of the same dataset to be read at the same time.
//!
//! The library is meant to be used in conjunction with the [bindings to the official HDF5
//! library](https://docs.rs/hdf5/0.7.0/hdf5/index.html).
//!
//! ## Usage
//!
//! Create an [index](idx::Index), then read the values:
//!
//! ```
//! use hidefix::prelude::*;
//!
//! let idx = Index::index("tests/data/coads_climatology.nc4").unwrap();
//! let mut r = idx.reader("SST").unwrap();
//!
//! let values = r.values::<f32, _>(..).unwrap();
//!
//! println!("SST: {:?}", values);
//! ```
//!
//! or convert a [hdf5::File] or [hdf5::Dataset] into an index by using
//! [`try_from`](std::convert::TryFrom) or the [`index`](idx::IntoIndex) method.
//!
//! or use the [IntoIndex](idx::IntoIndex) trait:
//!
//! ```
//! use hidefix::prelude::*;
//!
//! let i = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap().index().unwrap();
//! let iv = i.reader("SST").unwrap().values::<f32, _>(..).unwrap();
//! ```
//!
//! ## NetCDF4 files
//!
//! NetCDF4 uses HDF5 as their underlying data-format. Hidefix can be used to read the NetCDF
//! variables, though there might be extra decoding necessary. The hidefix-`xarray` does that for
//! you in the python bindings.
//!
//! ```
//! use std::convert::TryInto;
//! use hidefix::prelude::*;
//!
//! let f = netcdf::open("tests/data/coads_climatology.nc4").unwrap();
//! let nv = f.variable("SST").unwrap().values::<f32, _>(..).unwrap();
//!
//! let i: Index = (&f).try_into().unwrap();
//! let iv = i.reader("SST").unwrap().values::<f32, _>(..).unwrap();
//!
//! assert_eq!(iv, nv);
//! ```
//!
//! or use the [IntoIndex](idx::IntoIndex) trait:
//!
//! ```
//! use hidefix::prelude::*;
//!
//! let i = netcdf::open("tests/data/coads_climatology.nc4").unwrap().index().unwrap();
//! let iv = i.reader("SST").unwrap().values::<f32, _>(..).unwrap();
//! ```
//!
//! It is also possible to [stream](reader::stream::StreamReader) the values. The streamer is
//! currently optimized for streaming bytes.
//!
//! ## Fast indexing
//!
//! The indexing can be sped up considerably (_about 200x_) by using the new interface to [iterating
//! over chunks](https://github.com/HDFGroup/hdf5/pull/6) in HDF5. The `fast-index` feature flag currently requires a patched version of
//! [hdf5-rust](https://github.com/magnusuMET/hdf5-rust/tree/hidefix_jul_2023). You therefore have to use `patch` to
//! point the `hdf5` and `hdf5-sys` dependencies to the patched versions for now, in your
//! `Cargo.toml`:
//!
//! ```ignore
//! [patch.crates-io]
//! hdf5 = { git = "https://github.com/magnusuMET/hdf5-rust", branch = "hidefix_jul_2023" }
//! hdf5-sys = { git = "https://github.com/magnusuMET/hdf5-rust", branch = "hidefix_jul_2023" }
//! hdf5-src = { git = "https://github.com/magnusuMET/hdf5-rust", branch = "hidefix_jul_2023" }
//! ```

#![allow(incomplete_features)]
#![recursion_limit = "1024"]
#![feature(test)]
#![feature(cow_is_borrowed, array_methods)]
#![feature(assert_matches)]
#![feature(slice_group_by)]
#![feature(mutex_unlock)]
#![feature(new_uninit)]

extern crate test;

pub mod extent;
pub mod filters;
pub mod idx;
pub mod reader;

pub mod prelude {
    pub use super::extent::{Extent, Extents};
    pub use super::idx::{DatasetExt, Datatype, Index, IntoIndex};
    pub use super::reader::{ParReader, ParReaderExt, Reader, ReaderExt, Streamer, StreamerExt};
}

#[cfg(feature = "python")]
pub mod python;
