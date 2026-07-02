//! Reader fetching chunks from S3 (or any S3-compatible object store).
//!
//! Chunks are fetched with ranged `GET`s: requests are made asynchronously (several in
//! flight at a time) while decompression is done synchronously as the chunks arrive.
//! Chunks that are (close to) adjacent in the file are coalesced into a single request.
//!
//! The [`Bucket`] is configured by the caller (region, endpoint, credentials), the reader
//! only issues ranged `GET`s for the object key against it:
//!
//! ```no_run
//! use hidefix::prelude::*;
//! use hidefix::idx::DatasetD;
//! use hidefix::reader::s3::S3Reader;
//!
//! let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
//! let DatasetD::D3(ds) = i.dataset("SST").unwrap() else { panic!() };
//!
//! let bucket = s3::Bucket::new(
//!     "my-bucket",
//!     "eu-west-1".parse().unwrap(),
//!     s3::creds::Credentials::default().unwrap(),
//! ).unwrap();
//! let mut r = S3Reader::with_dataset(ds, bucket, "coads_climatology.nc4").unwrap();
//!
//! let values = r.values::<f32, _>(..).unwrap();
//! ```
use std::future::Future;

use anyhow::ensure;
use futures::{StreamExt, TryStreamExt};
use s3::Bucket;

use super::{chunk::decode_chunk, dataset::Reader};
use crate::extent::Extents;
use crate::filters::byteorder::Order;
use crate::idx::{Chunk, Dataset};

/// Maximum number of concurrent range requests.
const CONCURRENT_REQUESTS: usize = 16;

/// Chunks closer together than this (bytes) are coalesced into a single range request,
/// the gap is fetched and discarded.
const COALESCE_GAP: u64 = 32 * 1024;

/// Maximum size (bytes) of a coalesced range request.
const MAX_REQUEST_SZ: u64 = 8 * 1024 * 1024;

pub struct S3Reader<'a, const D: usize> {
    ds: &'a Dataset<'a, D>,
    bucket: Box<Bucket>,
    key: String,
    chunk_sz: u64,
}

/// Segments of a chunk: destination offset, start and end (in the same units as
/// [`Dataset::group_chunk_slices`]).
type Segments = Vec<(u64, u64, u64)>;

/// A single ranged `GET` covering one or more chunks.
struct Request<'a, const D: usize> {
    /// Byte address of first chunk.
    start: u64,

    /// One past the last byte of the last chunk.
    end: u64,

    /// Chunks in request with their segments.
    chunks: Vec<(&'a Chunk<D>, Segments)>,
}

/// Group chunk slices (sorted by file address, as returned by
/// [`Dataset::group_chunk_slices`]) into coalesced range requests.
fn group_requests<'a, const D: usize>(
    groups: Vec<(&'a Chunk<D>, u64, u64, u64)>,
) -> Vec<Request<'a, D>> {
    let mut requests: Vec<Request<'a, D>> = Vec::new();

    for (c, current, start, end) in groups {
        let addr = c.addr.get();
        let sz = c.size.get();

        match requests.last_mut() {
            Some(r) if r.chunks.last().unwrap().0.addr == c.addr => {
                // Another slice of the previous chunk.
                r.chunks.last_mut().unwrap().1.push((current, start, end));
            }
            Some(r)
                if addr >= r.end
                    && addr - r.end <= COALESCE_GAP
                    && (addr + sz) - r.start <= MAX_REQUEST_SZ =>
            {
                // Close enough to the previous chunk: coalesce into the same request.
                r.end = addr + sz;
                r.chunks.push((c, vec![(current, start, end)]));
            }
            _ => requests.push(Request {
                start: addr,
                end: addr + sz,
                chunks: vec![(c, vec![(current, start, end)])],
            }),
        }
    }

    requests
}

/// Drive a future to completion from a blocking context, whether or not we are already
/// inside a tokio runtime.
fn block_on<F: Future + Send>(fut: F) -> F::Output
where
    F::Output: Send,
{
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    let runtime = || RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().unwrap());

    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(fut))
        }
        // `block_in_place` panics on a current-thread runtime: drive the future on the
        // fallback runtime from a scoped thread instead.
        Ok(_) => std::thread::scope(|scope| {
            scope
                .spawn(|| runtime().block_on(fut))
                .join()
                .expect("block_on thread panicked")
        }),
        Err(_) => runtime().block_on(fut),
    }
}

impl<'a, const D: usize> S3Reader<'a, D> {
    pub fn with_dataset<S: Into<String>>(
        ds: &'a Dataset<D>,
        bucket: Box<Bucket>,
        key: S,
    ) -> Result<S3Reader<'a, D>, anyhow::Error> {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(S3Reader {
            ds,
            bucket,
            key: key.into(),
            chunk_sz,
        })
    }

    /// Reads raw bytes of slice into destination buffer, async version of
    /// [`Reader::read_to`]. Range requests are made concurrently, decompression is done
    /// synchronously as the chunks arrive.
    pub async fn read_to_async(
        &self,
        extents: &Extents,
        dst: &mut [u8],
    ) -> Result<usize, anyhow::Error> {
        let dsz = self.ds.dsize as u64;
        let counts = extents.get_counts(&self.ds.shape)?;
        let vsz = counts.product::<u64>() * dsz;

        ensure!(
            dst.len() >= vsz as usize,
            "destination buffer has insufficient capacity"
        );

        async fn fetch<'r, const D: usize>(
            bucket: &Bucket,
            key: &str,
            r: Request<'r, D>,
        ) -> Result<(Request<'r, D>, bytes::Bytes), anyhow::Error> {
            let resp = bucket
                .get_object_range(key, r.start, Some(r.end - 1))
                .await?;
            ensure!(
                resp.status_code() == 206 || resp.status_code() == 200,
                "range request failed: status code: {}",
                resp.status_code()
            );
            let bytes = resp.into_bytes();
            ensure!(
                bytes.len() as u64 >= r.end - r.start,
                "range request returned too few bytes: {} < {}",
                bytes.len(),
                r.end - r.start
            );
            Ok((r, bytes))
        }

        let requests = group_requests(self.ds.group_chunk_slices(extents))
            .into_iter()
            .map(|r| fetch(&self.bucket, &self.key, r))
            .collect::<Vec<_>>();

        let mut responses = futures::stream::iter(requests).buffer_unordered(CONCURRENT_REQUESTS);

        while let Some((request, bytes)) = responses.try_next().await? {
            for (c, segments) in request.chunks {
                let offset = (c.addr.get() - request.start) as usize;
                let chunk = bytes[offset..(offset + c.size.get() as usize)].to_vec();

                let cache = decode_chunk(
                    chunk,
                    self.chunk_sz,
                    dsz,
                    self.ds.gzip.is_some(),
                    self.ds.shuffle,
                )?;

                for (current, start, end) in segments {
                    let start = (start * dsz) as usize;
                    let end = (end * dsz) as usize;
                    let current = (current * dsz) as usize;

                    debug_assert!(start <= end);
                    debug_assert!(end <= cache.len());

                    dst[current..(current + (end - start))].copy_from_slice(&cache[start..end]);
                }
            }
        }

        Ok(vsz as usize)
    }
}

impl<const D: usize> Reader for S3Reader<'_, D> {
    fn order(&self) -> Order {
        self.ds.order
    }

    fn dsize(&self) -> usize {
        self.ds.dsize
    }

    fn shape(&self) -> &[u64] {
        &self.ds.shape
    }

    fn read_to(&mut self, extents: &Extents, dst: &mut [u8]) -> Result<usize, anyhow::Error> {
        block_on(self.read_to_async(extents, dst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks() -> Vec<Chunk<1>> {
        // contiguous, gap of 1k, gap > COALESCE_GAP
        vec![
            Chunk::new(1000, 500, [0]),
            Chunk::new(1500, 500, [10]),
            Chunk::new(2500, 500, [20]),
            Chunk::new(3000 + 2 * COALESCE_GAP, 500, [30]),
        ]
    }

    #[test]
    fn group_coalesced() {
        let chunks = chunks();
        let groups = chunks.iter().map(|c| (c, 0, 0, 10)).collect::<Vec<_>>();

        let requests = group_requests(groups);
        assert_eq!(requests.len(), 2);

        // chunk 0 and 1 are contiguous, chunk 2 is within COALESCE_GAP.
        assert_eq!(requests[0].start, 1000);
        assert_eq!(requests[0].end, 3000);
        assert_eq!(requests[0].chunks.len(), 3);

        assert_eq!(requests[1].start, 3000 + 2 * COALESCE_GAP);
        assert_eq!(requests[1].chunks.len(), 1);
    }

    #[test]
    fn group_slices_of_same_chunk() {
        let chunks = chunks();
        let groups = vec![
            (&chunks[0], 0, 0, 10),
            (&chunks[0], 10, 20, 30),
            (&chunks[1], 20, 0, 10),
        ];

        let requests = group_requests(groups);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].chunks.len(), 2);
        assert_eq!(requests[0].chunks[0].1, vec![(0, 0, 10), (10, 20, 30)]);
    }

    #[test]
    fn group_max_request_size() {
        let chunks = [
            Chunk::new(1000, MAX_REQUEST_SZ - 500, [0]),
            Chunk::new(1000 + MAX_REQUEST_SZ - 500, 1000, [10]),
        ];
        let groups = chunks.iter().map(|c| (c, 0, 0, 10)).collect::<Vec<_>>();

        let requests = group_requests(groups);
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn group_empty() {
        let requests = group_requests(Vec::<(&Chunk<1>, u64, u64, u64)>::new());
        assert!(requests.is_empty());
    }
}
