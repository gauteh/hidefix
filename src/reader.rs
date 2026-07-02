pub mod cache;
pub(crate) mod chunk;
pub mod dataset;
pub mod direct;
#[cfg(feature = "s3")]
pub mod s3;
pub mod stream;

pub use dataset::{ParReader, ParReaderExt, Reader, ReaderExt, Streamer, StreamerExt};
