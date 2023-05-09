use itertools::izip;

use crate::idx::{Chunk, Dataset};

/// An iterator over chunk, start in chunk and end in chunk which if joined will make up the specified slice through the
/// variable.
pub struct ChunkSlice<'a, const D: usize> {
    dataset: &'a Dataset<'a, D>,

    slice_start: [u64; D],
    slice_counts: [u64; D],

    /// The iterators offset in values from the start of the slice.
    slice_offset: u64,

    /// The end of slice in values from the start of the slice.
    slice_end: u64,
}

impl<'a, const D: usize> ChunkSlice<'a, D> {
    /// Empty slice returned for indices and counts that are out of bounds or of zero size.
    pub fn empty(dataset: &'a Dataset<D>) -> ChunkSlice<'a, D> {
        ChunkSlice {
            dataset,
            slice_start: [0; D],
            slice_counts: [0; D],
            slice_offset: 0,
            slice_end: 0,
        }
    }

    pub fn new(dataset: &'a Dataset<D>, indices: [u64; D], counts: [u64; D]) -> ChunkSlice<'a, D> {
        // Number of values to return.
        let slice_end = if dataset.is_scalar() {
            // scalar
            assert!(indices.is_empty());
            1
        } else {
            // size of slice dimensions
            counts.iter().product::<u64>()
        };

        debug_assert_eq!(indices.len(), dataset.shape.len());
        debug_assert_eq!(counts.len(), dataset.shape.len());

        // Check that slice is within bounds of Dataset. Also checked in `Dataset::chunk_slices`.
        debug_assert!(izip!(&indices, &counts, &dataset.shape).all(|(i, c, z)| i + c <= *z));

        let slice_start = indices;
        let slice_counts = counts;
        let slice_offset = 0;

        ChunkSlice {
            dataset,
            slice_start,
            slice_counts,
            slice_offset,
            slice_end,
        }
    }
}
impl<'a, const D: usize> Iterator for ChunkSlice<'a, D> {
    type Item = (&'a Chunk<D>, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.slice_offset >= self.slice_end {
            debug_assert!(self.slice_offset <= self.slice_end);
            return None;
        }

        // Scalar value dataset: Just return the first chunk.
        if self.dataset.is_scalar() {
            debug_assert!(self.dataset.chunks.len() == 1);
            debug_assert!(self.dataset.shape.is_empty());
            self.slice_offset += 1;

            return Some((&self.dataset.chunks[0], 0, 1));
        }

        // Advance as far as possible in the current chunk:
        //
        // 1. Start at the last dimension
        //
        // 2. If the slice contains the entire dimension, and all lower dimensions,
        //    continue to the next.
        //
        // 3. If the slice only contains a dimension partially, stop and yield the slice.
        //
        // 2D:
        //
        // |------------|------------|------------|------------|
        // |s...........|......      |            |            |
        // |............|.....e      |            |            |
        // |            |            |            |            |
        // |------------|------------|------------|------------|
        //
        //
        // 3D:
        //
        // |-------------|-------------|-------------|-------------|
        // |s.....|......|......|      |      |      |      |      |
        // |......|......|.....e|      |      |      |      |      |
        // |------|------|------|------|------|------|------|------|
        // |      |      |      |      |      |      |      |      |
        // |      |      |      |      |      |      |      |      |
        // |-------------|-------------|-------------|-------------|
        //
        //
        //

        // let slice_offset_coords = offset_to_coords(self.slice_offset

        // Elements to advance in current sub-slice of slice.
        let mut advance = 0;

        for di in (0..D).rev() {
            let current_offset = self.slice_offset + advance;

            let dim_sz = self.dataset.shape[di];
        }

        // let ds_offset =
        None
    }
}

fn coords_to_offset<const D: usize>(coords: [u64; D], dim_sz: [u64; D]) -> u64 {
    coords.iter().zip(dim_sz).map(|(c, sz)| c * sz).sum()
}

/// Convert an offset in shape to coordinates.
fn offset_to_coords<const D: usize>(offset: u64, dim_sz: [u64; D]) -> [u64; D] {
    dim_sz
        .iter()
        .scan(offset, |offset, sz| {
            let c = *offset / sz;
            *offset = *offset - (c * sz);

            Some(c)
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::filters::byteorder::Order as ByteOrder;

    pub(crate) fn test_dataset() -> Dataset<'static, 2> {
        Dataset::new(
            Datatype::Float(2),
            ByteOrder::BE,
            [20, 20],
            vec![
                Chunk::new(0, 400, [0, 0]),
                Chunk::new(400, 400, [0, 10]),
                Chunk::new(800, 400, [10, 0]),
                Chunk::new(1200, 400, [10, 10]),
            ],
            [10, 10],
            false,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_offset_to_coords() {
        assert_eq!(offset_to_coords(0, [2, 1, 1]), [0, 0, 0]);
        assert_eq!(offset_to_coords(1, [2, 1, 1]), [0, 1, 0]);
        assert_eq!(offset_to_coords(2, [2, 1, 1]), [1, 0, 0]);

        assert_eq!(offset_to_coords(0, [8, 4, 1]), [0, 0, 0]);
        assert_eq!(offset_to_coords(2, [8, 4, 1]), [0, 0, 2]);
        assert_eq!(offset_to_coords(4, [8, 4, 1]), [0, 1, 0]);
        assert_eq!(offset_to_coords(16+4, [8, 4, 1]), [2, 1, 0]);

        assert_eq!(coords_to_offset([2, 1, 0], [8, 4, 1]), 16+4);
        assert_eq!(coords_to_offset([0, 1, 0], [8, 4, 1]), 4);
        assert_eq!(coords_to_offset([0, 0, 2], [8, 4, 1]), 2);
    }

    #[test]
    fn chunk_slice_1() {
        let chunks = (0..31)
            .map(|i| Chunk::new(i * 16, 16, [i]))
            .collect::<Vec<_>>();

        let ds = Dataset::new(
            Datatype::UInt(4),
            ByteOrder::BE,
            [31],
            chunks,
            [1],
            false,
            None,
        )
        .unwrap();

        ChunkSlice::new(&ds, [0], [31]).for_each(drop);
        ChunkSlice::new(&ds, [0], [4]).for_each(drop);
    }

    #[test]
    fn chunk_slice_11n() {
        let chunks = (0..2)
            .map(|i| (0..32).map(move |j| Chunk::new(i * 32 + j * 1000, 635000, [i, j, 0])))
            .flatten()
            .collect::<Vec<_>>();

        let ds = Dataset::new(
            Datatype::Int(2),
            ByteOrder::BE,
            [2, 32, 580],
            chunks,
            [1, 1, 580],
            false,
            None,
        )
        .unwrap();

        // ds.chunk_slices(None, Some(&[2, 4, 580, 1202]))
        //     .for_each(drop);
        ChunkSlice::new(&ds, [0, 0, 0], [2, 32, 580]).for_each(drop);
    }
}
