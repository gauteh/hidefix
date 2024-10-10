//! Extents used for putting and getting data
//! from a variable

use std::convert::Infallible;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use anyhow::Result;

#[derive(Debug, Clone, Copy)]
/// An extent of a dimension
///
/// This enum has many ways to be constructed using `TryFrom`:
/// ```rust
/// # use hidefix::extent::Extent;
/// fn take_extent(e: impl TryInto<Extent>) {}
/// take_extent(3);
/// take_extent(..);
/// take_extent(..5);
/// take_extent(..=5);
/// take_extent(3..);
/// // Start at 3 up to the 74th index
/// take_extent(3..74);
/// take_extent(3..=74);
/// // Start at 3 with 74 elements
/// take_extent((3, 74));
/// ```
pub enum Extent {
    /// A slice
    Slice {
        /// Start of slice
        start: u64,
    },
    /// A slice with a given end
    SliceEnd {
        /// Start of slice
        start: u64,
        /// End of slice
        end: u64,
    },
    /// A slice with a given count
    SliceCount {
        /// Start of slice
        start: u64,
        /// Number of elements in slice
        count: u64,
    },
    /// A slice which is just an index
    Index(u64),
}

macro_rules! impl_for_ref {
    ($from: ty : $item: ty) => {
        impl From<&$from> for $item {
            fn from(e: &$from) -> Self {
                Self::from(e.clone())
            }
        }
    };
    (TryFrom $from: ty : $item: ty) => {
        impl TryFrom<&$from> for $item {
            type Error = anyhow::Error;
            fn try_from(e: &$from) -> Result<Self, Self::Error> {
                Self::try_from(e.clone())
            }
        }
    };
}

impl From<u64> for Extent {
    fn from(start: u64) -> Self {
        Self::Index(start)
    }
}
impl_for_ref!(u64: Extent);

impl From<RangeFrom<u64>> for Extent {
    fn from(range: RangeFrom<u64>) -> Self {
        Self::Slice { start: range.start }
    }
}
impl_for_ref!(RangeFrom<u64> : Extent);

impl From<Range<u64>> for Extent {
    fn from(range: Range<u64>) -> Self {
        Self::SliceEnd {
            start: range.start,
            end: range.end,
        }
    }
}
impl_for_ref!(Range<u64> : Extent);

impl From<RangeTo<u64>> for Extent {
    fn from(range: RangeTo<u64>) -> Self {
        Self::SliceEnd {
            start: 0,
            end: range.end,
        }
    }
}
impl_for_ref!(RangeTo<u64> : Extent);

impl From<RangeToInclusive<u64>> for Extent {
    fn from(range: RangeToInclusive<u64>) -> Self {
        Self::SliceEnd {
            start: 0,
            end: range.end + 1,
        }
    }
}
impl_for_ref!(RangeToInclusive<u64> : Extent);

impl From<RangeInclusive<u64>> for Extent {
    fn from(range: RangeInclusive<u64>) -> Self {
        Self::SliceEnd {
            start: *range.start(),
            end: range.end() + 1,
        }
    }
}
impl_for_ref!(RangeInclusive<u64> : Extent);

impl From<RangeFull> for Extent {
    fn from(_: RangeFull) -> Self {
        Self::Slice { start: 0 }
    }
}
impl_for_ref!(RangeFull: Extent);

impl From<(u64, u64)> for Extent {
    fn from((start, count): (u64, u64)) -> Self {
        Self::SliceCount { start, count }
    }
}
impl_for_ref!((u64, u64): Extent);

#[derive(Debug, Clone, Default)]
/// A selector for getting data in a dataset
///
/// This type can be constructed in many ways
/// ```rust
/// # use hidefix::extent::{Extent, Extents};
/// fn take_extents(extents: impl TryInto<Extents>) {}
/// // Get all values
/// take_extents(..);
/// // Get array with only first 10 of the first dimension
/// // and the first 2 of the second dimension
/// take_extents([..10, ..2]);
/// // Get values after some index
/// take_extents([1.., 2..]);
/// // The above syntax (using arrays) does not allow arbitrary types
/// // for each `Extent`, for this use tuples
/// take_extents((
///     1..10,
///     (2..=100),
///     4,
///     (3, 4),
/// ));
/// // Or specify counts using slices of `Extent`
/// take_extents([
///     Extent::SliceCount { start: 0, count: 10 },
///     (5..).into(),
/// ]);
/// // One can use two arrays to specify start and count separately
/// take_extents((&[1, 2, 3], &[3, 2, 1]));
/// // The `ndarray::s!` macro can also be used
/// take_extents(ndarray::s![3, 5..]);
/// ```
pub enum Extents {
    /// The full variable
    #[default]
    All,
    /// A selection along each dimension
    Extent(Vec<Extent>),
}

impl From<std::ops::RangeFull> for Extents {
    fn from(_: std::ops::RangeFull) -> Self {
        Self::All
    }
}

impl From<Vec<Extent>> for Extents {
    fn from(slice: Vec<Extent>) -> Self {
        Self::Extent(slice)
    }
}

impl From<&'_ [Extent]> for Extents {
    fn from(slice: &[Extent]) -> Self {
        Self::Extent(slice.to_owned())
    }
}

impl<const N: usize> From<[Extent; N]> for Extents {
    fn from(slice: [Extent; N]) -> Self {
        Self::Extent(slice.to_vec())
    }
}

macro_rules! impl_extent_as_extents {
    ($item: ty) => {
        impl From<$item> for Extents {
            fn from(item: $item) -> Self {
                Self::from(&item)
            }
        }

        impl From<&$item> for Extents {
            fn from(item: &$item) -> Self {
                Self::Extent(vec![item.into()])
            }
        }
    };
    (TryFrom $item: ty) => {
        impl TryFrom<$item> for Extents {
            type Error = anyhow::Error;
            fn try_from(item: $item) -> Result<Self, Self::Error> {
                Ok(Self::Extent(vec![item.try_into()?]))
            }
        }
        impl TryFrom<&$item> for Extents {
            type Error = anyhow::Error;
            fn try_from(item: &$item) -> Result<Self, Self::Error> {
                Ok(Self::Extent(vec![item.clone().try_into()?]))
            }
        }
    };
}

impl_extent_as_extents!(u64);
impl_extent_as_extents!(RangeFrom<u64>);
impl_extent_as_extents!(Range<u64>);
impl_extent_as_extents!(RangeTo<u64>);
impl_extent_as_extents!(RangeToInclusive<u64>);
impl_extent_as_extents!(RangeInclusive<u64>);

macro_rules! impl_extent_arrlike {
    ($item: ty) => {
        impl From<&'_ [$item]> for Extents {
            fn from(slice: &[$item]) -> Self {
                Self::Extent(slice.iter().map(|s| s.into()).collect())
            }
        }
        impl From<Vec<$item>> for Extents {
            fn from(slice: Vec<$item>) -> Self {
                Self::from(slice.as_slice())
            }
        }

        impl<const N: usize> From<[$item; N]> for Extents {
            fn from(slice: [$item; N]) -> Self {
                Self::from(slice.as_slice())
            }
        }
        impl<const N: usize> From<&[$item; N]> for Extents {
            fn from(slice: &[$item; N]) -> Self {
                Self::from(slice.as_slice())
            }
        }
    };
    (TryFrom $item: ty) => {
        impl TryFrom<&'_ [$item]> for Extents
        //where <$item as TryInto<Extent>>::Error: Into<anyhow::Error>,
        {
            type Error = anyhow::Error;
            fn try_from(slice: &[$item]) -> Result<Self, Self::Error> {
                Ok(Self::Extent(
                    slice
                        .iter()
                        .map(|s| {
                            let extent: Extent = s.try_into()?;
                            Ok(extent)
                        })
                        .collect::<Result<Vec<Extent>, anyhow::Error>>()?,
                ))
            }
        }
        impl TryFrom<Vec<$item>> for Extents {
            type Error = anyhow::Error;
            fn try_from(slice: Vec<$item>) -> Result<Self, Self::Error> {
                Self::try_from(slice.as_slice())
            }
        }

        impl<const N: u64> TryFrom<[$item; N]> for Extents {
            type Error = anyhow::Error;
            fn try_from(slice: [$item; N]) -> Result<Self, Self::Error> {
                Self::try_from(slice.as_slice())
            }
        }
        impl<const N: u64> TryFrom<&[$item; N]> for Extents {
            type Error = anyhow::Error;
            fn try_from(slice: &[$item; N]) -> Result<Self, Self::Error> {
                Self::try_from(slice.as_slice())
            }
        }
    };
}

impl_extent_arrlike!(u64);
impl_extent_arrlike!(RangeFrom<u64>);
impl_extent_arrlike!(Range<u64>);
impl_extent_arrlike!(RangeTo<u64>);
impl_extent_arrlike!(RangeToInclusive<u64>);
impl_extent_arrlike!(RangeInclusive<u64>);
impl_extent_arrlike!(RangeFull);
impl_extent_arrlike!((u64, u64));

macro_rules! impl_tuple {
    () => ();

    ($head:ident, $($tail:ident,)*) => (
        #[allow(non_snake_case)]
        impl<$head, $($tail,)*> TryFrom<($head, $($tail,)*)> for Extents
            where
                $head: TryInto<Extent>,
                $head::Error: Into<anyhow::Error>,
                $(
                    $tail: TryInto<Extent>,
                    $tail::Error: Into<anyhow::Error>,
                )*
        {
            type Error = anyhow::Error;
            fn try_from(slice: ($head, $($tail,)*)) -> Result<Self, Self::Error> {
                let ($head, $($tail,)*) = slice;
                Ok(vec![($head).try_into().map_err(|e| e.into())?, $(($tail).try_into().map_err(|e| e.into())?,)*].into())
            }
        }

        impl_tuple! { $($tail,)* }
    )
}

impl_tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, }

impl From<()> for Extents {
    fn from(_: ()) -> Self {
        Self::Extent(vec![])
    }
}

impl Extent {
    /// Starting index
    fn start(&self) -> u64 {
        match *self {
            Self::Index(idx) => idx,
            Self::Slice { start }
            | Self::SliceCount { start, count: _ }
            | Self::SliceEnd { start, end: _ } => start,
        }
    }
    /// Number of elements along this extent
    fn count(&self) -> Option<u64> {
        match *self {
            Self::Index(_) => Some(1),
            Self::Slice { start: _ } => None,
            Self::SliceCount { start: _, count } => Some(count),
            Self::SliceEnd { start, end } => Some((start..end).count() as u64),
        }
    }

    /// Make into an `Extent` which has a known count
    fn canonicalise(&self, dimsize: u64) -> Self {
        match *self {
            Self::Index(start) => Extent::Index(start),
            Self::Slice { start } => Extent::SliceCount {
                start,
                count: (start..dimsize).count() as u64,
            },
            Self::SliceEnd { start, end } => Extent::SliceCount {
                start,
                count: (start..end).count() as u64,
            },
            Self::SliceCount { start, count } => Extent::SliceCount { start, count },
        }
    }
}

/// Iterator which creates canonicalised `Extent`s (where count is always `Some`)
enum ExtentIterator<'a> {
    All(std::slice::Iter<'a, u64>),
    Extents(std::iter::Zip<std::slice::Iter<'a, Extent>, std::slice::Iter<'a, u64>>),
}

impl Iterator for ExtentIterator<'_> {
    type Item = Extent;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::All(iter) => iter
                .next()
                .map(|&count| Extent::SliceCount { start: 0, count }),
            Self::Extents(iter) => iter.next().map(|(&extent, &d)| extent.canonicalise(d)),
        }
    }
}

impl DoubleEndedIterator for ExtentIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::All(iter) => iter
                .next_back()
                .map(|&count| Extent::SliceCount { start: 0, count }),
            Self::Extents(iter) => iter.next_back().map(|(&extent, &d)| extent.canonicalise(d)),
        }
    }
}

pub(crate) type StartCount = (Vec<u64>, Vec<u64>);

impl Extents {
    /// Transform Extents into a series of `Extent` which all have a
    /// `count` which returns `Some`
    pub(crate) fn canonicalise<'a>(
        &'a self,
        dims: &'a [u64],
    ) -> Result<impl DoubleEndedIterator<Item = Extent> + 'a, anyhow::Error> {
        match self {
            Extents::All => Ok(ExtentIterator::All(dims.iter())),
            Extents::Extent(extents) => {
                if extents.len() != dims.len() {
                    return Err(anyhow::anyhow!(
                        "Extents had length {} but dimension length is {}",
                        extents.len(),
                        dims.len(),
                    ));
                }
                Ok(ExtentIterator::Extents(extents.iter().zip(dims.iter())))
            }
        }
    }
    /// Get sizes along the dims
    pub(crate) fn get_counts<'a>(
        &'a self,
        dims: &'a [u64],
    ) -> Result<impl Iterator<Item = u64> + 'a, anyhow::Error> {
        Ok(self.canonicalise(dims)?.map(|e| e.count().unwrap()))
    }
    /// Get both starting index and sizes
    pub(crate) fn get_start_count(&self, dims: &[u64]) -> Result<StartCount, anyhow::Error> {
        Ok(self
            .canonicalise(dims)?
            .map(|e| (e.start(), e.count().unwrap()))
            .unzip())
    }
    /// Same as `get_start_count`, but errors if `D` is not compatible
    pub(crate) fn get_start_count_sized<const D: usize>(
        &self,
        dims: &[u64; D],
    ) -> Result<([u64; D], [u64; D]), anyhow::Error> {
        let (start, count) = self.get_start_count(dims)?;
        anyhow::ensure!(start.len() == D, "shape is not compatible with extent");
        Ok((start.try_into().unwrap(), count.try_into().unwrap()))
    }
}

mod ndarray_impl {
    use super::*;
    use ndarray::{Dimension, SliceInfo, SliceInfoElem};

    impl<T, Din: Dimension, Dout: Dimension> TryFrom<&'_ SliceInfo<T, Din, Dout>> for Extents
    where
        T: AsRef<[SliceInfoElem]>,
    {
        type Error = anyhow::Error;
        fn try_from(slice: &SliceInfo<T, Din, Dout>) -> Result<Self, Self::Error> {
            let slice: &[SliceInfoElem] = slice.as_ref();

            Ok(slice
                .iter()
                .map(|&s| match s {
                    SliceInfoElem::Slice { start, end, step } => {
                        let start = u64::try_from(start)?;
                        if step != 1 {
                            Err(anyhow::anyhow!("Strides are not supported"))
                        } else if let Some(end) = end {
                            let end = u64::try_from(end)?;
                            Ok(Extent::SliceEnd { start, end })
                        } else {
                            Ok(Extent::Slice { start })
                        }
                    }
                    SliceInfoElem::Index(index) => {
                        let index =
                            u64::try_from(index).map_err(|_| anyhow::anyhow!("Invalid index"))?;
                        Ok(Extent::Index(index))
                    }
                    SliceInfoElem::NewAxis => {
                        Err(anyhow::anyhow!("Can't add new axis in this context"))
                    }
                })
                .collect::<Result<Vec<Extent>, Self::Error>>()?
                .into())
        }
    }

    impl<T, Din: Dimension, Dout: Dimension> TryFrom<SliceInfo<T, Din, Dout>> for Extents
    where
        T: AsRef<[SliceInfoElem]>,
    {
        type Error = anyhow::Error;
        fn try_from(slice: SliceInfo<T, Din, Dout>) -> Result<Self, Self::Error> {
            Self::try_from(&slice)
        }
    }
}

impl TryFrom<(&[u64], &[u64])> for Extents {
    type Error = anyhow::Error;
    fn try_from((start, count): (&[u64], &[u64])) -> Result<Self, Self::Error> {
        if start.len() == count.len() {
            Ok(Self::Extent(
                start
                    .iter()
                    .zip(count)
                    .map(|(&start, &count)| Extent::SliceCount { start, count })
                    .collect(),
            ))
        } else {
            Err(anyhow::anyhow!(
                "Indices and count does not have the same length"
            ))
        }
    }
}

impl TryFrom<(Vec<u64>, Vec<u64>)> for Extents {
    type Error = anyhow::Error;
    fn try_from((start, count): (Vec<u64>, Vec<u64>)) -> Result<Self, Self::Error> {
        Self::try_from((start.as_slice(), count.as_slice()))
    }
}

macro_rules! impl_extents_for_arrays {
    ($N: expr) => {
        impl TryFrom<([u64; $N], [u64; $N])> for Extents {
            type Error = Infallible;
            fn try_from((start, count): ([u64; $N], [u64; $N])) -> Result<Self, Self::Error> {
                    Self::try_from((&start, &count))
            }
        }

        impl TryFrom<(&[u64; $N], &[u64; $N])> for Extents {
            type Error = Infallible;
            fn try_from((start, count): (&[u64; $N], &[u64; $N])) -> Result<Self, Self::Error> {
                    Ok(Self::Extent(
                        start
                            .iter()
                            .zip(count)
                            .map(|(&start, &count)| Extent::SliceCount {
                                start,
                                count,
                            })
                            .collect(),
                    ))
            }
        }
    };
    ($($N: expr,)*) => {
        $(impl_extents_for_arrays! { $N })*
    };
}
impl_extents_for_arrays! { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, }

impl From<&Self> for Extents {
    fn from(extents: &Self) -> Self {
        extents.clone()
    }
}

#[cfg(test)]
mod test {
    use super::{Extent, Extents};
    use anyhow::Result;
    fn take_extent<E>(e: E) -> Result<Extent>
    where
        E: TryInto<Extent>,
        E::Error: Into<anyhow::Error>,
    {
        e.try_into().map_err(|e| e.into())
    }

    fn take_extents<E>(e: E) -> Result<Extents>
    where
        E: TryInto<Extents>,
        E::Error: Into<anyhow::Error>,
    {
        e.try_into().map_err(|e| e.into())
    }

    #[test]
    fn test_extent() -> Result<()> {
        let _ = take_extent(1)?;
        let _ = take_extent(1..)?;
        let _ = take_extent(1..5)?;
        let _ = take_extent(..5)?;
        let _ = take_extent(..=5)?;
        let _ = take_extent(4..=5)?;

        // start+count
        let _ = take_extent((5, 4))?;

        // Empty slice
        let _ = take_extent(1..0)?;
        let _ = take_extent(1..=1)?;
        let _ = take_extent(1..=2)?;

        Ok(())
    }

    #[test]
    fn test_extents() -> Result<()> {
        // This is the "All" type
        let extent = take_extents(..)?;
        match extent {
            Extents::All => {}
            _ => panic!(),
        }

        // These are for 1D specs
        let _ = take_extents(1)?;
        let _ = take_extents(1..)?;
        let _ = take_extents(1..5)?;
        let _ = take_extents(..5)?;
        let _ = take_extents(..=5)?;
        let _ = take_extents(4..=5)?;

        // These are multidimensional

        // Array
        let _ = take_extents([.., ..])?;
        let _ = take_extents([1, 2])?;
        let _ = take_extents([1.., 2..])?;
        let _ = take_extents([1..5, 2..6])?;
        let _ = take_extents([..5, ..6])?;
        let _ = take_extents([..=5, ..=6])?;
        let _ = take_extents([4..=50, 5..=8])?;

        // Slice
        let _ = take_extents([.., ..].as_slice())?;
        let _ = take_extents([1, 2].as_slice())?;
        let _ = take_extents([1.., 2..].as_slice())?;
        let _ = take_extents([1..5, 2..6].as_slice())?;
        let _ = take_extents([..5, ..6].as_slice())?;
        let _ = take_extents([..=5, ..=6].as_slice())?;
        let _ = take_extents([4..=5, 5..=6].as_slice())?;

        // Vec
        let _ = take_extents(vec![.., ..])?;
        let _ = take_extents(vec![1, 2])?;
        let _ = take_extents(vec![1.., 2..])?;
        let _ = take_extents(vec![1..5, 2..6])?;
        let _ = take_extents(vec![..5, ..6])?;
        let _ = take_extents(vec![..=5, ..=6])?;
        let _ = take_extents(vec![4..=5, 5..=6])?;

        // Tuple
        let _ = take_extents((1_u64.., 2_u64))?;
        let _ = take_extents((1.., 2))?;
        let _ = take_extents((1.., 2, ..6))?;

        let _ = take_extents(ndarray::s![2..5, 4])?;
        let _ = take_extents(ndarray::s![2..;4, 4]).unwrap_err();

        // (start, count)
        let _ = take_extents(([1, 2], [3, 4]))?;
        let _ = take_extents(([1, 2].as_slice(), [3, 4].as_slice()))?;
        let _ = take_extents((&[1, 2], &[3, 4]))?;
        let _ = take_extents((vec![1, 2], vec![3, 4]))?;

        // [(s0, c0), (s1, c1)]
        let _ = take_extents([(1, 2), (3, 4)])?;
        let _ = take_extents([(1, 2), (3, 4)].as_slice())?;
        let _ = take_extents(&[(1, 2), (3, 4)])?;
        let _ = take_extents(vec![(1, 2), (3, 4)])?;

        // Use of borrowed Extents
        let e: Extents = (..).into();
        let _ = take_extents(&e)?;
        let _ = take_extents(e)?;

        Ok(())
    }
}
