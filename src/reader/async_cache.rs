use std::pin::Pin;
use std::cmp;
use std::io::Read;
use std::sync::Arc;
use std::mem;
use futures::ready;
use futures::task::{Context, Poll};
use futures::io::{self, AsyncRead, AsyncSeek, SeekFrom};

use lru::LruCache;

use crate::filters;
use crate::idx::Dataset;

pub struct DatasetReader<R>
where R: AsyncRead + AsyncSeek + Unpin + Send + Sync + 'static + Unpin
{
    dsz: u64,
    chunk_slices: std::vec::IntoIter<(u64, u64, u64, u64)>,
    fd: Pin<Box<R>>,
    cache: LruCache<u64, Arc<Vec<u8>>>,
    chunk_sz: u64,
    gzip: bool,
    shuffle: bool,
    state: St,
    size: usize
}

impl<R> DatasetReader<R>
    where
        R: AsyncRead + AsyncSeek + Unpin + Send + Sync + 'static + Unpin
{
    pub fn with_dataset_read(
        ds: &Dataset,
        fd: R,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<DatasetReader<R>, anyhow::Error> {
        const CACHE_SZ: u64 = 32 * 1024 * 1024;
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;
        let cache_sz = cmp::max(CACHE_SZ / chunk_sz, 1);

        let counts: &[u64] = counts.unwrap_or_else(|| ds.shape.as_slice());
        let chunk_slices = ds.chunk_slices(indices, Some(&counts))
                                .map(|(c, start, end)| (c.addr, c.size, start, end))
                                .collect::<Vec<(u64, u64, u64, u64)>>()
                                .into_iter();

        let dsz = ds.dsize as u64;
        let size = counts.iter().product::<u64>() * dsz;
        let gzip = ds.gzip.is_some();
        let shuffle = ds.shuffle;

        let fd = Box::pin(fd);

        Ok(DatasetReader {
            dsz,
            chunk_slices,
            fd,
            cache: LruCache::new(cache_sz as usize),
            chunk_sz,
            gzip,
            shuffle,
            state: St::PendingChunk,
            size: size as usize
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

pub enum St {
    PendingChunk,
    SeekChunk { chunk: (u64, u64, u64, u64) },
    ReadChunk { chunk: (u64, u64, u64, u64) },
    ConsumeChunk { chunk: (u64, u64, u64, u64), bytes: Arc<Vec<u8>>, pos: u64 },
    Eof,
}

impl<R> AsyncRead for DatasetReader<R>
    where R: AsyncRead + AsyncSeek + Unpin + Send + Sync + 'static
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        loop {
            match &mut self.state {
                St::PendingChunk => {
                    match self.chunk_slices.next() {
                        Some((addr, size, start, end)) => {
                            match self.cache.get(&addr) {
                                Some(bytes) => {
                                    self.state = St::ConsumeChunk {
                                        chunk: (addr, size, start, end),
                                        bytes: Arc::clone(bytes),
                                        pos: 0
                                    };
                                },
                                None => {
                                    self.state = St::SeekChunk {
                                        chunk: (addr, size, start, end)
                                    };
                                }
                            }
                        },

                        None => {
                            self.state = St::Eof;
                            return Poll::Ready(Ok(0));
                        }
                    }
                },

                St::SeekChunk { chunk } => {
                    let chunk = *chunk;
                    match ready!(self.fd.as_mut().poll_seek(cx, SeekFrom::Start(chunk.0))) {
                        Ok(p) => {
                            assert!(p == chunk.0);
                            self.state = St::ReadChunk { chunk };
                        },
                        Err(e) => return Poll::Ready(Err(e))
                    }
                },

                St::ReadChunk { chunk } => {
                    let chunk = *chunk;
                    let mut bytes = vec![0_u8; chunk.1 as usize];
                    let mut buf = bytes.as_mut_slice();

                    // from io::AsyncReadExt::ReadExact
                    while !buf.is_empty() {
                        let n = ready!(self.fd.as_mut().poll_read(cx, buf))?;
                        {
                            let (_, rest) = mem::replace(&mut buf, &mut []).split_at_mut(n);
                            buf = rest;
                        }
                        if n == 0 {
                            return Poll::Ready(Err(io::ErrorKind::UnexpectedEof.into()))
                        }
                    }

                    // Decompress
                    let cache = if self.gzip {
                        let mut decache: Vec<u8> = Vec::with_capacity(self.chunk_sz as usize);
                        unsafe {
                            decache.set_len(self.chunk_sz as usize);
                        }

                        let mut dz = flate2::read::ZlibDecoder::new_with_buf(
                            &bytes[..],
                            vec![0_u8; 32 * 1024 * 1024],
                        );
                        dz.read_exact(&mut decache)?;

                        decache
                    } else {
                        bytes
                    };

                    let cache = if self.shuffle {
                        filters::shuffle::unshuffle_sized(&cache, self.dsz as usize)
                    } else {
                        cache
                    };

                    let cache = Arc::new(cache);
                    self.cache.put(chunk.0, Arc::clone(&cache));
                    self.state = St::ConsumeChunk { chunk, bytes: cache, pos: 0 };
                },

                St::ConsumeChunk {chunk, bytes, mut pos } => {
                    let (_, _, start, end) = *chunk;
                    let bytes = Arc::clone(&bytes);

                    let dsz = self.dsz;
                    let start = (start * dsz) as usize;
                    let end = (end * dsz) as usize;

                    let b0 = start + pos as usize;
                    let b1 = cmp::min(b0 + buf.len(), end);
                    let nread = b1 - b0;

                    buf[..nread].copy_from_slice(&bytes[b0..b1]);

                    pos += nread as u64;

                    if pos >= end as u64 {
                        self.state = St::PendingChunk;
                    }

                    return Poll::Ready(Ok(nread))
                }

                St::Eof => return Poll::Ready(Ok(0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::Index;
    use futures::executor::block_on;
    use futures::io::AsyncReadExt;
    use async_std::fs;
    use crate::filters::byteorder::ToNative;
    use byte_slice_cast::IntoVecOf;

    #[test]
    fn read_coads_sst() {
        block_on(async {
            let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
            let ds = i.dataset("SST").unwrap();
            let fd = fs::File::open(i.path().unwrap()).await.unwrap();
            let mut r = DatasetReader::with_dataset_read(ds, fd, None, None).unwrap();
            // let br = BufReader::new(r);

            let mut buf = Vec::with_capacity(r.size());
            r.read_to_end(&mut buf).await.unwrap();
            let mut vs = buf.into_vec_of::<f32>().unwrap();
            vs.to_native(ds.order);


            let h = hdf5::File::open(i.path().unwrap()).unwrap();
            let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

            assert_eq!(vs, hvs);
        })
    }
}
