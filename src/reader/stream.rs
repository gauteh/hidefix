use async_stream::stream;
use futures::stream::{Stream, StreamExt};
use futures_util::pin_mut;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};

use byte_slice_cast::{FromByteVec, IntoVecOf};

use crate::idx::Dataset;

pub struct DatasetReader<'a> {
    ds: &'a Dataset,
    p: PathBuf,
}

impl<'a> DatasetReader<'a> {
    pub fn with_dataset<P>(ds: &'a Dataset, p: P) -> Result<DatasetReader<'a>, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        Ok(DatasetReader {
            ds,
            p: p.as_ref().into(),
        })
    }

    pub fn stream(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> impl Stream<Item = Result<Vec<u8>, anyhow::Error>> {
        let dsz = self.ds.dtype.size() as u64;

        let counts: &[u64] = counts.unwrap_or(self.ds.shape.as_slice());
        let slices = self
            .ds
            .chunk_slices(indices, Some(&counts))
            .map(|(c, a, b)| (c.addr + a * dsz, c.addr + b * dsz))
            .collect::<Vec<_>>();

        let p = self.p.clone();

        stream! {
            let mut fd = File::open(p)?;

            for (start, end) in slices {
                let slice_sz = (end - start) as usize;

                let mut buf = Vec::with_capacity(slice_sz);
                unsafe {
                    buf.set_len(slice_sz);
                }

                fd.seek(SeekFrom::Start(start))?;
                fd.read_exact(&mut buf)?;

                yield Ok(buf)
            }
        }
    }

    pub fn stream_values<T>(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> impl Stream<Item = Result<Vec<T>, anyhow::Error>>
    where
        T: FromByteVec + Unpin,
    {
        // TODO: BE, LE conversion
        // TODO: use as_slice_of() to avoid copy, or possible values_to(&mut buf) so that
        //       caller keeps ownership of slice too.
        let reader = self.stream(indices, counts);

        stream! {
            pin_mut!(reader);
            while let Some(b) = reader.next().await {
                yield match b {
                    Ok(b) => b.into_vec_of::<T>().map_err(|_| anyhow!("could not cast to value")),
                    Err(e) => Err(e)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::Index;
    use futures::executor::block_on_stream;

    #[test]
    fn read_t_float32() {
        let i = Index::index("tests/data/t_float.h5").unwrap();
        let r = DatasetReader::with_dataset(i.dataset("d32_1").unwrap(), i.path()).unwrap();

        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d32_1").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_1d() {
        let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
        let r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_2d() {
        let i = Index::index("tests/data/chunked_twoD.h5").unwrap();
        let r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }
}
