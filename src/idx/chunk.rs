use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A HDF5 chunk. A chunk is read and written in its entirety by the HDF5 library. This is
/// usually necessary since the chunk can be compressed and filtered.
///
/// > Note: The official HDF5 library uses a 1MB dataset cache by default.
///
/// [HDF5 chunking](https://support.hdfgroup.org/HDF5/doc/Advanced/Chunking/index.html).
#[derive(Debug, Eq, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub addr: u64,
    pub offset: Vec<u64>,

    /// Chunk size in bytes (storage size)
    pub size: u64,
}

impl Chunk {
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
}

impl Hash for Chunk {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.addr.hash(state);
    }
}

impl Ord for Chunk {
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

impl PartialOrd for Chunk {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Chunk {
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
                offset: vec![10, 0, 0],
                size: 5,
                addr: 5,
            },
            Chunk {
                offset: vec![0, 0, 0],
                size: 10,
                addr: 50,
            },
            Chunk {
                offset: vec![10, 10, 0],
                size: 1,
                addr: 1,
            },
            Chunk {
                offset: vec![0, 10, 0],
                size: 1,
                addr: 1,
            },
            Chunk {
                offset: vec![0, 0, 10],
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
            offset: vec![10, 10],
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
}
