use crate::filters::byteorder::ToNative;
use byte_slice_cast::{FromByteVec, IntoVecOf};

pub trait Reader {
    fn read(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<u8>, anyhow::Error>;

    fn values<T>(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<T>, anyhow::Error>
    where
        T: FromByteVec,
        [T]: ToNative;
}

pub enum UnifyReader<T1, T2, T3>
where
    T1: Reader,
    T2: Reader,
    T3: Reader,
{
    R1(T1),
    R2(T2),
    R3(T3),
}

impl<T1, T2, T3> Reader for UnifyReader<T1, T2, T3>
where
    T1: Reader,
    T2: Reader,
    T3: Reader,
{
    fn read(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        match self {
            Self::R1(r) => r.read(indices, counts),
            Self::R2(r) => r.read(indices, counts),
            Self::R3(r) => r.read(indices, counts),
        }
    }

    fn values<T>(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<T>, anyhow::Error>
    where
        T: FromByteVec,
        [T]: ToNative,
    {
        match self {
            Self::R1(r) => r.values(indices, counts),
            Self::R2(r) => r.values(indices, counts),
            Self::R3(r) => r.values(indices, counts),
        }
    }
}

use super::stream::StreamReader;

type S1<'a> = StreamReader<'a, 1>;
type S2<'a> = StreamReader<'a, 2>;
type S3<'a> = StreamReader<'a, 3>;

pub enum UnifyStreamer<'a> {
    R1(S1<'a>),
    R2(S2<'a>),
    R3(S3<'a>),
}

use futures::{Stream, StreamExt};
use bytes::Bytes;
use crate::filters::byteorder::{self, Order};

impl<'a> UnifyStreamer<'a> {
    /// A stream of bytes from the variable. Always in Big Endian.
    pub fn stream(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> impl Stream<Item = Result<Bytes, anyhow::Error>> {
        match self {
            Self::R1(st) => st.stream(indices, counts),
            _ => unimplemented!()
        }
    }

    pub fn order(&self) -> Order {
        match self {
            Self::R1(st) => st.order(),
            _ => unimplemented!()
        }
    }

    pub fn stream_values<T>(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> impl Stream<Item = Result<Vec<T>, anyhow::Error>>
    where
        T: FromByteVec + Unpin,
        [T]: ToNative,
    {
        match self {
            Self::R1(st) => st.stream_values(indices, counts),
            _ => unimplemented!()
        }
    }
}
