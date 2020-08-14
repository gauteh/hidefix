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

pub enum UnifyReader<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
where
    T0: Reader,
    T1: Reader,
    T2: Reader,
    T3: Reader,
    T4: Reader,
    T5: Reader,
    T6: Reader,
    T7: Reader,
    T8: Reader,
    T9: Reader,
{
    R0(T0),
    R1(T1),
    R2(T2),
    R3(T3),
    R4(T4),
    R5(T5),
    R6(T6),
    R7(T7),
    R8(T8),
    R9(T9),
}

impl<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> Reader for UnifyReader<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
where
    T0: Reader,
    T1: Reader,
    T2: Reader,
    T3: Reader,
    T4: Reader,
    T5: Reader,
    T6: Reader,
    T7: Reader,
    T8: Reader,
    T9: Reader,
{
    fn read(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        match self {
            Self::R0(r) => r.read(indices, counts),
            Self::R1(r) => r.read(indices, counts),
            Self::R2(r) => r.read(indices, counts),
            Self::R3(r) => r.read(indices, counts),
            Self::R4(r) => r.read(indices, counts),
            Self::R5(r) => r.read(indices, counts),
            Self::R6(r) => r.read(indices, counts),
            Self::R7(r) => r.read(indices, counts),
            Self::R8(r) => r.read(indices, counts),
            Self::R9(r) => r.read(indices, counts),
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
            Self::R0(r) => r.values(indices, counts),
            Self::R1(r) => r.values(indices, counts),
            Self::R2(r) => r.values(indices, counts),
            Self::R3(r) => r.values(indices, counts),
            Self::R4(r) => r.values(indices, counts),
            Self::R5(r) => r.values(indices, counts),
            Self::R6(r) => r.values(indices, counts),
            Self::R7(r) => r.values(indices, counts),
            Self::R8(r) => r.values(indices, counts),
            Self::R9(r) => r.values(indices, counts),
        }
    }
}

use super::stream::StreamReader;

type S0<'a> = StreamReader<'a, 0>;
type S1<'a> = StreamReader<'a, 1>;
type S2<'a> = StreamReader<'a, 2>;
type S3<'a> = StreamReader<'a, 3>;
type S4<'a> = StreamReader<'a, 4>;
type S5<'a> = StreamReader<'a, 5>;
type S6<'a> = StreamReader<'a, 6>;
type S7<'a> = StreamReader<'a, 7>;
type S8<'a> = StreamReader<'a, 8>;
type S9<'a> = StreamReader<'a, 9>;

pub enum UnifyStreamer<'a> {
    R0(S0<'a>),
    R1(S1<'a>),
    R2(S2<'a>),
    R3(S3<'a>),
    R4(S4<'a>),
    R5(S5<'a>),
    R6(S6<'a>),
    R7(S7<'a>),
    R8(S8<'a>),
    R9(S9<'a>),
}

use crate::filters::byteorder::{self, Order};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use async_stream::stream;
use futures::pin_mut;

impl<'a> UnifyStreamer<'a> {
    /// A stream of bytes from the variable. Always in Big Endian.
    pub fn stream(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
        ) -> impl Stream<Item = Result<Bytes, anyhow::Error>> {
        let mut boxed = match self {
            Self::R0(st) => st.stream(indices, counts).boxed(),
            Self::R1(st) => st.stream(indices, counts).boxed(),
            Self::R2(st) => st.stream(indices, counts).boxed(),
            Self::R3(st) => st.stream(indices, counts).boxed(),
            Self::R4(st) => st.stream(indices, counts).boxed(),
            Self::R5(st) => st.stream(indices, counts).boxed(),
            Self::R6(st) => st.stream(indices, counts).boxed(),
            Self::R7(st) => st.stream(indices, counts).boxed(),
            Self::R8(st) => st.stream(indices, counts).boxed(),
            Self::R9(st) => st.stream(indices, counts).boxed(),
        };

        stream! {
            while let Some(v) = boxed.next().await {
                yield v;
            }
        }
    }

    pub fn order(&self) -> Order {
        match self {
            Self::R0(st) => st.order(),
            Self::R1(st) => st.order(),
            Self::R2(st) => st.order(),
            Self::R3(st) => st.order(),
            Self::R4(st) => st.order(),
            Self::R5(st) => st.order(),
            Self::R6(st) => st.order(),
            Self::R7(st) => st.order(),
            Self::R8(st) => st.order(),
            Self::R9(st) => st.order(),
        }
    }

    pub fn stream_values<T>(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> impl Stream<Item = Result<Vec<T>, anyhow::Error>>
    where
        T: FromByteVec + Unpin + Send + 'static,
        [T]: ToNative,
    {
        let mut boxed = match self {
            Self::R0(st) => st.stream_values(indices, counts).boxed(),
            Self::R1(st) => st.stream_values(indices, counts).boxed(),
            Self::R2(st) => st.stream_values(indices, counts).boxed(),
            Self::R3(st) => st.stream_values(indices, counts).boxed(),
            Self::R4(st) => st.stream_values(indices, counts).boxed(),
            Self::R5(st) => st.stream_values(indices, counts).boxed(),
            Self::R6(st) => st.stream_values(indices, counts).boxed(),
            Self::R7(st) => st.stream_values(indices, counts).boxed(),
            Self::R8(st) => st.stream_values(indices, counts).boxed(),
            Self::R9(st) => st.stream_values(indices, counts).boxed(),
        };

        stream! {
            while let Some(v) = boxed.next().await {
                yield v;
            }
        }
    }
}
