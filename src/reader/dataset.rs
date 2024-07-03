use anyhow::ensure;
use byte_slice_cast::{AsMutByteSlice, AsSliceOf, FromByteSlice, ToMutByteSlice};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::pin::Pin;

use crate::extent::Extents;
use crate::filters::byteorder::{Order, ToNative};

pub trait Reader {
    /// Reads raw bytes of slice into destination buffer. Returns bytes read.
    fn read_to(&mut self, extents: &Extents, dst: &mut [u8]) -> Result<usize, anyhow::Error>;

    /// Byte-order of dataset.
    fn order(&self) -> Order;

    /// Size of datatype.
    fn dsize(&self) -> usize;

    /// Shape of dataset.
    fn shape(&self) -> &[u64];
}

#[cfg(feature = "unstable")]
fn empty_vec<T>(vsz: usize) -> Vec<T>
where
    T: Default,
{
    let values = Box::<[T]>::new_uninit_slice(vsz);
    let values = unsafe { values.assume_init() };
    values.into_vec()
}

#[cfg(not(feature = "unstable"))]
fn empty_vec<T>(vsz: usize) -> Vec<T>
where
    T: Default,
{
    let mut values = Vec::with_capacity(vsz);
    values.resize_with(vsz, T::default);
    values
}

pub trait ReaderExt: Reader {
    /// Reads values into desitination slice. Returns values read.
    fn values_to<T, E>(&mut self, extents: E, dst: &mut [T]) -> Result<usize, anyhow::Error>
    where
        T: ToMutByteSlice,
        [T]: ToNative,
        E: TryInto<Extents>,
        E::Error: Into<anyhow::Error>,
    {
        let extents = extents.try_into().map_err(|e| e.into())?;
        let r = self.read_to(&extents, dst.as_mut_byte_slice())?;
        dst.to_native(self.order());

        Ok(r)
    }

    /// Reads slice of dataset into `Vec<T>`.
    fn values<T, E>(&mut self, extents: E) -> Result<Vec<T>, anyhow::Error>
    where
        T: ToMutByteSlice + Default,
        [T]: ToNative,
        E: TryInto<Extents>,
        E::Error: Into<anyhow::Error>,
    {
        let dsz = self.dsize();
        let extents = extents.try_into().map_err(|e| e.into())?;
        let counts = extents.get_counts(self.shape())?;
        let vsz = counts.product::<u64>() as usize * dsz / std::mem::size_of::<T>();

        ensure!(
            dsz % std::mem::size_of::<T>() == 0,
            "size of datatype ({}) not multiple of target {}",
            dsz,
            std::mem::size_of::<T>()
        );

        ensure!((dsz * vsz) % std::mem::align_of::<T>() == 0, "alignment of datatype ({}) not a multiple of datatype size and length {}*{}={}, alignment may not match and result in unsoundness", std::mem::align_of::<T>(), dsz, vsz, vsz * dsz);

        let mut values = empty_vec(vsz);
        self.values_to(extents, values.as_mut_slice())?; // XXX: take maybeuninit

        Ok(values)
    }
}

impl<T: ?Sized + Reader> ReaderExt for T {}

pub trait ParReader {
    fn read_to_par(&self, extents: &Extents, dst: &mut [u8]) -> Result<usize, anyhow::Error>;
}

pub trait ParReaderExt: Reader + ParReader {
    /// Reads values into desitination slice. Returns values read.
    fn values_to_par<T, E>(&self, extents: E, dst: &mut [T]) -> Result<usize, anyhow::Error>
    where
        T: ToMutByteSlice,
        [T]: ToNative,
        E: TryInto<Extents>,
        E::Error: Into<anyhow::Error>,
    {
        let extents = extents.try_into().map_err(|e| e.into())?;
        let r = self.read_to_par(&extents, dst.as_mut_byte_slice())?;
        dst.to_native(self.order());

        Ok(r)
    }

    /// Reads slice of dataset into `Vec<T>`.
    fn values_par<T, E>(&self, extents: E) -> Result<Vec<T>, anyhow::Error>
    where
        T: ToMutByteSlice + Default,
        [T]: ToNative,
        E: TryInto<Extents>,
        E::Error: Into<anyhow::Error>,
    {
        let dsz = self.dsize();
        let extents = extents.try_into().map_err(|e| e.into())?;
        let counts = extents.get_counts(self.shape())?;
        let vsz = counts.product::<u64>() as usize * dsz / std::mem::size_of::<T>();

        ensure!(
            dsz % std::mem::size_of::<T>() == 0,
            "size of datatype ({}) not multiple of target {}",
            dsz,
            std::mem::size_of::<T>()
        );

        ensure!((dsz * vsz) % std::mem::align_of::<T>() == 0, "alignment of datatype ({}) not a multiple of datatype size and length {}*{}={}, alignment may not match and result in unsoundness", std::mem::align_of::<T>(), dsz, vsz, vsz * dsz);

        let mut values = empty_vec(vsz);
        self.values_to_par(extents, values.as_mut_slice())?;

        Ok(values)
    }

    fn values_dyn_par<T, E>(&self, extents: E) -> Result<ndarray::ArrayD<T>, anyhow::Error>
    where
        T: ToMutByteSlice + Default,
        [T]: ToNative,
        E: TryInto<Extents>,
        E::Error: Into<anyhow::Error>,
    {
        let dsz = self.dsize();
        let extents = extents.try_into().map_err(|e| e.into())?;
        let counts = extents.get_counts(self.shape())?;
        let dims = counts.map(|d| d as usize).collect::<Vec<_>>();
        let vsz = dims.iter().product::<usize>() * dsz / std::mem::size_of::<T>();

        ensure!(
            dsz % std::mem::size_of::<T>() == 0,
            "size of datatype ({}) not multiple of target {}",
            dsz,
            std::mem::size_of::<T>()
        );

        ensure!((dsz * vsz) % std::mem::align_of::<T>() == 0, "alignment of datatype ({}) not a multiple of datatype size and length {}*{}={}, alignment may not match and result in unsoundness", std::mem::align_of::<T>(), dsz, vsz, vsz * dsz);

        // this is not safe: better to let read_to take maybeuninit's
        let mut a = unsafe { ndarray::ArrayD::<T>::uninit(dims).assume_init() };
        let dst = a.as_slice_mut().unwrap();
        self.values_to_par(extents, dst)?;

        Ok(a)
    }
}

impl<T: ?Sized + Reader + ParReader> ParReaderExt for T {}

pub trait Streamer {
    /// Stream slice of dataset as chunks of `Bytes`.
    fn stream(
        &self,
        extents: &Extents,
    ) -> Pin<Box<dyn Stream<Item = Result<Bytes, anyhow::Error>> + Send + 'static>>;

    /// Stream slice of dataset as chunks of `Bytes` serialized as XDR/DAP2-DODS.
    fn stream_xdr(
        &self,
        extents: &Extents,
    ) -> Pin<Box<dyn Stream<Item = Result<Bytes, anyhow::Error>> + Send + 'static>>;

    /// Byte-order of dataset.
    fn order(&self) -> Order;

    /// Size of datatype.
    fn dsize(&self) -> usize;
}

pub trait StreamerExt: Streamer {
    /// Stream slice of dataset as `Vec<T>`.
    fn stream_values<T, E>(
        &self,
        extents: E,
    ) -> Pin<Box<dyn Stream<Item = Result<Vec<T>, anyhow::Error>> + Send + 'static>>
    where
        T: Unpin + Send + FromByteSlice + Clone,
        [T]: ToNative,
        E: TryInto<Extents>,
        E::Error: Into<anyhow::Error>,
    {
        let order = self.order();

        let extents = extents.try_into().map_err(|e| e.into()).unwrap();
        Box::pin(self.stream(&extents).map(move |b| {
            let b = b?;
            let values = b.as_slice_of::<T>()?;

            // Unfortunately we currently need to make a copy since the byte-slice may be
            // un-aligned to the slice of values.
            let mut values = values.to_vec();

            values.to_native(order);

            Ok(values)
        }))
    }
}

impl<T: ?Sized + Streamer> StreamerExt for T {}
