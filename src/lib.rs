//! # HIDEFIX
//!
//! A fast and concurrent reader for HDF5 and NetCDF (v4) files.
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
//! use hidefix::prelude::*;
//!
//! let idx = Index::index("tests/data/coads_climatology.nc4").unwrap();
//! let mut r = idx.reader("SST").unwrap();
//!
//! let values = r.values::<f32>(None, None).unwrap();
//!
//! println!("SST: {:?}", values);
//! ```
//!
//! or convert a [hdf5::File] or [hdf5::Dataset] into an index by using
//! [`try_from`](std::convert::TryFrom) or the [`index`](idx::IntoIndex) method.
//!
//!
//! It is also possible to [stream](reader::stream::StreamReader) the values. The streamer is
//! currently optimized for streaming bytes.
//!
//! ## Fast indexing
//!
//! The indexing can be sped up considerably (_about 200x_) by adding a new interface to iterating
//! over chunks in HDF5. The `fast-index` feature flag currently requires a patched version of
//! [hdf5-rust](https://github.com/gauteh/hdf5-rust/tree/hidefix) and
//! [hdf5](https://github.com/gauteh/hdf5/tree/chunk-iter-1-12). See this upstream
//! [pull-request](https://github.com/HDFGroup/hdf5/pull/6). You therefore have to use `patch` to
//! point the `hdf5` and `hdf5-sys` dependencies to the patched versions for now.

#![allow(incomplete_features)]
#![recursion_limit = "1024"]
#![feature(test)]
#![feature(cow_is_borrowed, array_methods)]
#![feature(assert_matches)]
#![feature(slice_group_by)]
#![feature(mutex_unlock)]
#![feature(new_uninit)]
extern crate test;

#[macro_use]
extern crate anyhow;

#[macro_use]
extern crate log;

pub mod filters;
pub mod idx;
pub mod reader;

pub mod prelude {
    pub use super::idx::{DatasetExt, Datatype, Index, IntoIndex};
    pub use super::reader::{ParReader, ParReaderExt, Reader, ReaderExt, Streamer, StreamerExt};
}

#[cfg(feature = "python")]
pub mod python;
