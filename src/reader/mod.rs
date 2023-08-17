pub mod cache;
pub(crate) mod chunk;
pub mod dataset;
pub mod direct;
pub mod stream;

#[cfg(feature = "s3")]
pub mod s3;

pub use dataset::{ParReader, ParReaderExt, Reader, ReaderExt, Streamer, StreamerExt};
