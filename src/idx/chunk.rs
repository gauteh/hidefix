use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A HDF5 chunk. A chunk is read and written in its entirety by the HDF5 library. This is
/// usually necessary since the chunk can be compressed and filtered.
///
/// > Note: The official HDF5 library uses a 1MB dataset cache by default.
///
/// HDF5 can store chunks in various types of data structures internally (`BTreeMap`, etc.), so
/// it is not necessarily a simple sorted array (presumably because chunks can be added at a later
/// time). The `get_chunk_info` methods iterate over this structure internally to get the requested
/// chunk (based on a predicate function set up internally). It would be far more efficient for us
/// if we could retrieve all chunks through one iteration, rather than do a full iteration for all
/// chunks which is obviously extremely inefficient -- and the main reason that indexing is slow.
///
/// Reference: [HDF5 chunking](https://support.hdfgroup.org/HDF5/doc/Advanced/Chunking/index.html).
#[derive(Debug, Eq, Clone)]
#[repr(C)]
pub struct Chunk<const D: usize>
where
    [u64; D]: std::array::LengthAtMost32,
{
    /// Address or offset (bytes) in file where chunk starts.
    pub addr: u64,

    /// Chunk size in bytes (storage size).
    pub size: u64,

    /// Coordinates of offset in dataspace where the chunk begins.
    pub offset: [u64; D],
}

impl<const D: usize> Chunk<D>
where
    [u64; D]: std::array::LengthAtMost32,
{
    /// Is the point described by the indices inside the chunk (`Equal`), before (`Less`) or after
    /// (`Greater`).
    #[must_use]
    pub fn contains(&self, i: &[u64], shape: &[u64]) -> Ordering {
        assert!(i.len() == shape.len());
        assert!(i.len() == self.offset.len());

        for j in 0..i.len() {
            if i[j] < self.offset[j] {
                return Ordering::Less;
            } else if i[j] >= self.offset[j] + shape[j] {
                return Ordering::Greater;
            }
        }

        Ordering::Equal
    }

    /// Reinterpret the Chunk as a slice of `u64`'s. This is ridiculously unsafe and I am not sure
    /// if I know what I am doing.
    ///
    /// Expressions in const-generics are not yet allowed.
    pub fn as_slice(&self) -> &[u64] {
        let ptr = self as *const Chunk<D>;
        let slice: &[u64] = unsafe {
            let ptr = std::mem::transmute(ptr);
            std::slice::from_raw_parts(ptr, std::mem::size_of::<Self>() / std::mem::size_of::<u64>())
        };

        assert_eq!(slice.len(), std::mem::size_of::<Self>() / std::mem::size_of::<u64>());

        slice
    }

    pub fn from_slice(slice: &[u64]) -> &Chunk<D> {
        assert_eq!(slice.len(), std::mem::size_of::<Self>() / std::mem::size_of::<u64>());

        unsafe {
            std::mem::transmute(slice.as_ptr())
        }
    }
}

impl<const D: usize> Hash for Chunk<D>
where
    [u64; D]: std::array::LengthAtMost32,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.addr.hash(state);
    }
}

impl<const D: usize> Ord for Chunk<D>
where
    [u64; D]: std::array::LengthAtMost32,
{
    fn cmp(&self, other: &Self) -> Ordering {
        for (aa, bb) in self.offset.iter().zip(&other.offset) {
            match aa.cmp(&bb) {
                Ordering::Greater => return Ordering::Greater,
                Ordering::Less => return Ordering::Less,
                Ordering::Equal => (),
            }
        }

        Ordering::Equal
    }
}

impl<const D: usize> PartialOrd for Chunk<D>
where
    [u64; D]: std::array::LengthAtMost32,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const D: usize> PartialEq for Chunk<D>
where
    [u64; D]: std::array::LengthAtMost32,
{
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering() {
        let mut v = vec![
            Chunk {
                offset: [10, 0, 0],
                size: 5,
                addr: 5,
            },
            Chunk {
                offset: [0, 0, 0],
                size: 10,
                addr: 50,
            },
            Chunk {
                offset: [10, 10, 0],
                size: 1,
                addr: 1,
            },
            Chunk {
                offset: [0, 10, 0],
                size: 1,
                addr: 1,
            },
            Chunk {
                offset: [0, 0, 10],
                size: 1,
                addr: 1,
            },
        ];
        v.sort();

        assert_eq!(v[0].offset, [0, 0, 0]);
        assert_eq!(v[1].offset, [0, 0, 10]);
        assert_eq!(v[2].offset, [0, 10, 0]);
        assert_eq!(v[3].offset, [10, 0, 0]);
        assert_eq!(v[4].offset, [10, 10, 0]);
    }

    #[test]
    fn contains() {
        let shape = [10, 10];

        let c = Chunk {
            offset: [10, 10],
            size: 10,
            addr: 0,
        };

        assert_eq!(c.contains(&[0, 0], &shape), Ordering::Less);
        assert_eq!(c.contains(&[5, 0], &shape), Ordering::Less);
        assert_eq!(c.contains(&[10, 0], &shape), Ordering::Less);
        assert_eq!(c.contains(&[10, 10], &shape), Ordering::Equal);
        assert_eq!(c.contains(&[5, 10], &shape), Ordering::Less);
        assert_eq!(c.contains(&[10, 15], &shape), Ordering::Equal);
        assert_eq!(c.contains(&[15, 15], &shape), Ordering::Equal);
        assert_eq!(c.contains(&[15, 10], &shape), Ordering::Equal);
        assert_eq!(c.contains(&[20, 20], &shape), Ordering::Greater);
        assert_eq!(c.contains(&[25, 20], &shape), Ordering::Greater);
        assert_eq!(c.contains(&[25, 10], &shape), Ordering::Greater);
        assert_eq!(c.contains(&[10, 25], &shape), Ordering::Greater);
        assert_eq!(c.contains(&[5, 25], &shape), Ordering::Less);
        assert_eq!(c.contains(&[25, 5], &shape), Ordering::Greater);
    }

    mod serde {
        use super::*;

        #[test]
        fn as_slice() {
            let c = Chunk {
                addr: 2,
                size: 7,
                offset: [10, 10],
            };

            let s = c.as_slice();
            println!("{:?} -> {:?}", c, s);
            assert_eq!(s, [2, 7, 10, 10]);

            // odd number of dims
            let c = Chunk {
                addr: 2,
                size: 7,
                offset: [10, 5, 10],
            };

            let s = c.as_slice();
            println!("{:?} -> {:?}", c, s);
            assert_eq!(s, [2, 7, 10, 5, 10]);

            let c = Chunk {
                addr: 2,
                size: 7,
                offset: [10, 5, 3, 10],
            };

            let s = c.as_slice();
            println!("{:?} -> {:?}", c, s);
            assert_eq!(s, [2, 7, 10, 5, 3, 10]);
        }

        #[test]
        fn from_slice() {
            let s = [2, 7, 10, 10];
            let c = Chunk::<2>::from_slice(&s);
            println!("{:?} -> {:?}", s, c);
            assert_eq!(c, &Chunk {
                addr: 2,
                size: 7,
                offset: [10, 10],
            });

            // odd number of dims
            let s = [2, 7, 10, 5, 10];
            let c = Chunk::<3>::from_slice(&s);
            println!("{:?} -> {:?}", s, c);
            assert_eq!(c, &Chunk {
                addr: 2,
                size: 7,
                offset: [10, 5, 10],
            });

            let s = [2, 7, 10, 5, 3, 10];
            let c = Chunk::<4>::from_slice(&s);
            println!("{:?} -> {:?}", s, c);
            assert_eq!(c, &Chunk {
                addr: 2,
                size: 7,
                offset: [10, 5, 3, 10],
            });
        }

        #[test]
        fn roundtrip() {
            let c = Chunk {
                addr: 2,
                size: 7,
                offset: [10, 5, 10],
            };

            let s = c.as_slice();
            assert_eq!(s, [2, 7, 10, 5, 10]);

            let c2 = Chunk::<3>::from_slice(s);

            assert_eq!(&c, c2);
        }
    }
}
