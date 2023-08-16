use std::cmp::Ordering;
use std::convert::TryInto;
use std::hash::{Hash, Hasher};

use byteorder::LittleEndian as LE;
use zerocopy::byteorder::U64;

pub type ULE = U64<LE>;

/// A HDF5 chunk. A chunk is read and written in its entirety by the HDF5 library. This is
/// usually necessary since the chunk can be compressed and filtered.
///
///
/// Reference: [HDF5 chunking](https://support.hdfgroup.org/HDF5/doc/Advanced/Chunking/index.html).
#[derive(Debug, Eq, Clone)]
#[repr(C)]
pub struct Chunk<const D: usize> {
    // WARNING: Do not alter repr, order or type of this struct or fields without verifying against
    //          slice operations.
    /// Address or offset (bytes) in file where chunk starts.
    pub addr: ULE,

    /// Chunk size in bytes (storage size).
    pub size: ULE,

    /// Coordinates of offset in dataspace where the chunk begins.
    pub offset: [ULE; D],
}

impl<const D: usize> Chunk<D> {
    pub fn new(addr: u64, size: u64, offset: [u64; D]) -> Chunk<D> {
        Chunk {
            addr: ULE::new(addr),
            size: ULE::new(size),
            offset: offset
                .iter()
                .cloned()
                .map(ULE::new)
                .collect::<Vec<_>>()
                .as_slice()
                .try_into()
                .unwrap(),
        }
    }

    pub fn offset_u64(&self) -> [u64; D] {
        self.offset
            .iter()
            .map(|o| o.get())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// Is the point described by the indices inside the chunk (`Equal`), before (`Less`) or after
    /// (`Greater`).
    #[must_use]
    pub fn contains(&self, i: &[u64], chunk_shape: &[u64]) -> Ordering {
        assert!(i.len() == chunk_shape.len());
        assert!(i.len() == self.offset.len());

        for j in 0..i.len() {
            if i[j] < self.offset[j].get() {
                return Ordering::Less;
            } else if i[j] >= self.offset[j].get() + chunk_shape[j] {
                return Ordering::Greater;
            }
        }

        Ordering::Equal
    }

    /// Reinterpret the Chunk as a slice of `u64`'s. This is ridiculously unsafe and I am not sure
    /// if I know what I am doing.
    ///
    /// Expressions in const-generics are not yet allowed.
    pub fn as_u64s(&self) -> &[ULE] {
        let ptr = self as *const Chunk<D>;
        let slice: &[ULE] = unsafe {
            let ptr = ptr as *const ULE;
            std::slice::from_raw_parts(
                ptr,
                std::mem::size_of::<Self>() / std::mem::size_of::<ULE>(),
            )
        };

        assert_eq!(
            slice.len(),
            std::mem::size_of::<Self>() / std::mem::size_of::<ULE>()
        );

        slice
    }

    /// Reinterpret a slice of `u64`s as a Chunk.
    pub fn from_u64s(slice: &[ULE]) -> &Chunk<D> {
        assert_eq!(
            slice.len(),
            std::mem::size_of::<Self>() / std::mem::size_of::<ULE>()
        );

        unsafe { &*(slice.as_ptr() as *const Chunk<D>) }
    }

    /// Reintepret a slice of `Chunk<D>`s to a slice of `u64`. This is efficient, but relies
    /// on unsafe code.
    pub fn slice_as_u64s(chunks: &[Chunk<D>]) -> &[ULE] {
        let ptr = chunks.as_ptr();
        let slice: &[ULE] = unsafe {
            let ptr = ptr as *const ULE;
            std::slice::from_raw_parts(
                ptr,
                std::mem::size_of_val(chunks) / std::mem::size_of::<ULE>(),
            )
        };

        assert_eq!(
            slice.len(),
            std::mem::size_of_val(chunks) / std::mem::size_of::<ULE>()
        );

        slice
    }

    /// Reintepret a slice of `u64`s to a slice of `Chunk<D>`. This is efficient, but relies
    /// on unsafe code.
    pub fn slice_from_u64s(slice: &[ULE]) -> &[Chunk<D>] {
        assert_eq!(
            slice.len() % (std::mem::size_of::<Self>() / std::mem::size_of::<ULE>()),
            0
        );
        let n = slice.len() / (std::mem::size_of::<Self>() / std::mem::size_of::<ULE>());

        let ptr = slice.as_ptr() as *const _;
        let chunks: &[Chunk<D>] = unsafe { std::slice::from_raw_parts(ptr, n) };

        assert_eq!(
            slice.len(),
            std::mem::size_of_val(chunks) / std::mem::size_of::<ULE>()
        );

        chunks
    }
}

impl<const D: usize> Hash for Chunk<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.addr.hash(state);
    }
}

impl<const D: usize> Ord for Chunk<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        for (aa, bb) in self.offset.iter().zip(&other.offset) {
            match aa.get().cmp(&bb.get()) {
                Ordering::Greater => return Ordering::Greater,
                Ordering::Less => return Ordering::Less,
                Ordering::Equal => (),
            }
        }

        Ordering::Equal
    }
}

impl<const D: usize> PartialOrd for Chunk<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const D: usize> PartialEq for Chunk<D> {
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment() {
        assert_eq!(std::mem::align_of::<Chunk<1>>(), 1);
        assert_eq!(std::mem::align_of::<Chunk<2>>(), 1);
        assert_eq!(std::mem::align_of::<Chunk<3>>(), 1);
        assert_eq!(std::mem::align_of::<Chunk<8>>(), 1);
    }

    #[test]
    fn ordering() {
        let mut v = vec![
            Chunk::new(5, 10, [10, 0, 0]),
            Chunk::new(50, 10, [0, 0, 0]),
            Chunk::new(1, 1, [10, 10, 0]),
            Chunk::new(1, 1, [0, 10, 0]),
            Chunk::new(1, 1, [0, 0, 10]),
        ];
        v.sort();

        assert_eq!(v[0].offset, [ULE::new(0), ULE::new(0), ULE::new(0)]);
        assert_eq!(v[1].offset, [ULE::new(0), ULE::new(0), ULE::new(10)]);
        assert_eq!(v[2].offset, [ULE::new(0), ULE::new(10), ULE::new(0)]);
        assert_eq!(v[3].offset, [ULE::new(10), ULE::new(0), ULE::new(0)]);
        assert_eq!(v[4].offset, [ULE::new(10), ULE::new(10), ULE::new(0)]);
    }

    #[test]
    fn contains() {
        let shape = [10, 10];

        let c = Chunk::new(0, 10, [10, 10]);
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
        use test::Bencher;

        #[test]
        fn as_u64s() {
            let c = Chunk::new(2, 7, [10, 10]);
            let s = c.as_u64s();
            println!("{:?} -> {:?}", c, s);
            assert_eq!(s, [ULE::new(2), ULE::new(7), ULE::new(10), ULE::new(10)]);

            // odd number of dims
            let c = Chunk::new(2, 7, [10, 5, 10]);
            let s = c.as_u64s();
            println!("{:?} -> {:?}", c, s);
            assert_eq!(
                s,
                [
                    ULE::new(2),
                    ULE::new(7),
                    ULE::new(10),
                    ULE::new(5),
                    ULE::new(10)
                ]
            );

            let c = Chunk::new(2, 7, [10, 5, 3, 10]);

            let s = c.as_u64s();
            println!("{:?} -> {:?}", c, s);
            assert_eq!(
                s,
                [
                    ULE::new(2),
                    ULE::new(7),
                    ULE::new(10),
                    ULE::new(5),
                    ULE::new(3),
                    ULE::new(10)
                ]
            );
        }

        #[test]
        fn from_u64s() {
            let s = [ULE::new(2), ULE::new(7), ULE::new(10), ULE::new(10)];
            let c = Chunk::<2>::from_u64s(&s);
            println!("{:?} -> {:?}", s, c);
            assert_eq!(c, &Chunk::new(2, 7, [10, 10]));

            // odd number of dims
            let s = [
                ULE::new(2),
                ULE::new(7),
                ULE::new(10),
                ULE::new(5),
                ULE::new(10),
            ];
            let c = Chunk::<3>::from_u64s(&s);
            println!("{:?} -> {:?}", s, c);
            assert_eq!(c, &Chunk::new(2, 7, [10, 5, 10]));

            let s = [
                ULE::new(2),
                ULE::new(7),
                ULE::new(10),
                ULE::new(5),
                ULE::new(3),
                ULE::new(10),
            ];
            let c = Chunk::<4>::from_u64s(&s);
            println!("{:?} -> {:?}", s, c);
            assert_eq!(c, &Chunk::new(2, 7, [10, 5, 3, 10]));
        }

        #[test]
        fn roundtrip() {
            let c = Chunk::new(2, 7, [10, 5, 10]);
            let s = c.as_u64s();
            assert_eq!(
                s,
                [
                    ULE::new(2),
                    ULE::new(7),
                    ULE::new(10),
                    ULE::new(5),
                    ULE::new(10)
                ]
            );

            let c2 = Chunk::<3>::from_u64s(s);

            assert_eq!(&c, c2);
        }

        #[test]
        fn slice_as_u64s() {
            let cs = [Chunk::new(2, 7, [10, 10]), Chunk::new(3, 17, [20, 20])];

            let s = Chunk::<2>::slice_as_u64s(&cs);
            println!("{:?} -> {:?}", cs, s);
            assert_eq!(
                s,
                [
                    ULE::new(2),
                    ULE::new(7),
                    ULE::new(10),
                    ULE::new(10),
                    ULE::new(3),
                    ULE::new(17),
                    ULE::new(20),
                    ULE::new(20)
                ]
            );

            let cs = [
                Chunk::new(2, 7, [10, 10, 15]),
                Chunk::new(3, 17, [20, 20, 30]),
            ];

            let s = Chunk::<3>::slice_as_u64s(&cs);
            println!("{:?} -> {:?}", cs, s);
            assert_eq!(
                s,
                [
                    ULE::new(2),
                    ULE::new(7),
                    ULE::new(10),
                    ULE::new(10),
                    ULE::new(15),
                    ULE::new(3),
                    ULE::new(17),
                    ULE::new(20),
                    ULE::new(20),
                    ULE::new(30)
                ]
            );
        }

        #[test]
        fn slice_from_u64s() {
            let s = [
                ULE::new(2),
                ULE::new(7),
                ULE::new(10),
                ULE::new(10),
                ULE::new(3),
                ULE::new(17),
                ULE::new(20),
                ULE::new(20),
            ];
            let cs = [Chunk::new(2, 7, [10, 10]), Chunk::new(3, 17, [20, 20])];

            let dcs = Chunk::<2>::slice_from_u64s(&s);
            println!("{:?} -> {:?}", s, dcs);
            assert_eq!(dcs, cs);

            let s = [
                ULE::new(2),
                ULE::new(7),
                ULE::new(10),
                ULE::new(10),
                ULE::new(15),
                ULE::new(3),
                ULE::new(17),
                ULE::new(20),
                ULE::new(20),
                ULE::new(30),
            ];
            let cs = [
                Chunk::new(2, 7, [10, 10, 15]),
                Chunk::new(3, 17, [20, 20, 30]),
            ];

            let dcs = Chunk::<3>::slice_from_u64s(&s);
            println!("{:?} -> {:?}", s, dcs);
            assert_eq!(dcs, cs);
        }

        #[bench]
        fn slice_from_u64s_10k_3d(b: &mut Bencher) {
            let chunks: Vec<Chunk<3>> = (0..10000)
                .map(|i| Chunk::new(i * 10, 300, [i * 10, i * 100, i * 10000]))
                .collect();

            assert_eq!(chunks.len(), 10000);

            let slice = Chunk::<3>::slice_as_u64s(chunks.as_slice());
            assert_eq!(
                slice.len(),
                10000 * std::mem::size_of::<Chunk<3>>() / std::mem::size_of::<u64>()
            );

            let dechunks = Chunk::<3>::slice_from_u64s(&slice);
            assert_eq!(dechunks.len(), chunks.len());
            assert_eq!(dechunks, chunks.as_slice());

            b.iter(|| {
                test::black_box(Chunk::<3>::slice_from_u64s(&slice));
            });
        }
    }
}
