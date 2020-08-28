//! # HIDEFIX
//!
//! A fast and concurrent reader for HDF5 and NetCDF (v4) files.
//!
//! > Currently requires Rust nightly.
//!
//! This library allows a HDF5 file to be read in a multi-threaded and concurrent way. The chunks
//! of a dataset need to be indexed in advance, this can be time-consuming, but efficient
//! serialization and partially zero-copy deserialization through [serde](https://serde.rs/) is
//! implemented. In particular by storing the indexes in a fast database, like
//! [sled](http://sled.rs/) allows speedy access.
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
//! use hidefix::idx::Index;
//! use hidefix::reader::Reader;
//!
//! let indx = Index::index("tests/data/coads_climatology.nc4").unwrap();
//! let mut r = indx.reader("SST").unwrap();
//!
//! let values = r.values::<f32>(None, None).unwrap();
//! println!("SST: {:?}", values);
//! ```
//!
//! It is also possible to [stream](reader::stream::StreamReader) the values. The streamer is
//! currently optimized for streaming bytes.

#![allow(incomplete_features)]
#![recursion_limit = "1024"]
#![feature(test)]
#![feature(const_generics, fixed_size_array, cow_is_borrowed)]
extern crate test;

#[macro_use]
extern crate anyhow;

pub mod filters;
pub mod idx;
pub mod reader;

pub use idx::IntoIndex;
