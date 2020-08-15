use byte_slice_cast::{
    AsByteSlice, AsMutByteSlice, FromByteVec, IntoByteVec, ToByteSlice, ToMutByteSlice,
};

use bytes::{Bytes, BytesMut};

/// Shuffle bytes according to HDF5 spec:
///
/// https://support.hdfgroup.org/ftp/HDF5//documentation/doc1.6/TechNotes/shuffling-algorithm-report.pdf
///
/// The shuffling algorithm re-arranges `bytes` with the following steps:
///
/// 1. put the first byte of each number in the first chunk
/// 2. put the second byte of each number in the second chunk
/// 3. repeat for size of number (e.g. 4 for i32).
///
/// Quoting the above:
///
/// For 5 32-bit integers: 1, 23, 43, 56, 35
///
/// they are laid out as following on a big-endian machine:
///
/// original: 0x00 0x00 0x00 0x01 0x00 0x00 0x00 0x17 0x00 0x00 0x00 0x2B 0x00 0x00 0x00 0x38 0x00 0x00 0x00 0x23
/// shuffled: 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x01 0x17 0x2B 0x38 0x23
///
/// Size of type `T` is used as word size.
pub fn shuffle<T, D>(src: &[T], dest: &mut [D])
where
    T: ToByteSlice,
    D: ToMutByteSlice,
{
    let sz = src.len();
    let wsz = std::mem::size_of::<T>();

    let src = src.as_byte_slice();
    let dest = dest.as_mut_byte_slice();

    assert!(dest.len() == src.len());

    for i in 0..wsz {
        for j in 0..sz {
            unsafe {
                *dest.get_unchecked_mut(i * sz + j) = *src.get_unchecked(j * wsz + i);
            }
        }
    }
}

/// Inverse of `shuffle`. Size of type `D` is used as word size.
pub fn unshuffle<T, D>(src: &[T], dest: &mut [D])
where
    T: ToByteSlice,
    D: ToMutByteSlice,
{
    let sz = dest.len();
    let wsz = std::mem::size_of::<D>();

    let src = src.as_byte_slice();
    let dest = dest.as_mut_byte_slice();

    assert!(dest.len() == src.len());

    for i in 0..wsz {
        for j in 0..sz {
            unsafe {
                *dest.get_unchecked_mut(j * wsz + i) = *src.get_unchecked(i * sz + j);
            }
        }
    }
}

/// Optimized unshuffle.
pub fn unshuffle_2<const N: usize>(src: &[u8], dest: &mut [u8])
{
    use std::convert::TryInto;

    assert!(src.len() == dest.len());
    assert!(src.len() % N == 0);
    let n = src.len() / N;

    let dest_ptr = dest.as_mut_ptr() as *mut _;
    let dest_structured: &mut [[u8; N]] = unsafe {
        std::slice::from_raw_parts_mut(dest_ptr, src.len() / N)
    };

    assert!(dest_structured.len() == dest.len() / N);

    dest_structured.iter_mut().enumerate().for_each(|(j, d)| {
        for i in 0..N {
            unsafe {
                *d.get_unchecked_mut(i) = *src.get_unchecked(i * n + j);
            }
        }
    })
}

/// Unshuffle bytes representing array with word size `wsz` (e.g. `4` for `int32`).
pub fn unshuffle_bytes(src: &Bytes, wsz: usize) -> BytesMut {
    debug_assert!(wsz > 1);
    assert_eq!(src.len() % wsz, 0);
    let sz = src.len() / wsz;

    let mut dest = BytesMut::with_capacity(src.len());
    unsafe {
        dest.set_len(src.len() as usize);
    }

    debug_assert_eq!(src.len(), dest.len());

    for i in 0..wsz {
        for j in 0..sz {
            unsafe {
                *dest.get_unchecked_mut(j * wsz + i) = *src.get_unchecked(i * sz + j);
            }
        }
    }

    dest
}

pub fn unshuffle_sized<T>(src: &[T], sz: usize) -> Vec<u8>
where
    T: ToByteSlice + FromByteVec + Copy,
{
    let src = src.as_byte_slice();
    let mut dest: Vec<u8> = Vec::with_capacity(src.len());
    unsafe { dest.set_len(src.len()) };

    match sz {
        1 => {
            // noop
            dest.copy_from_slice(&src);
        }
        2 => {
            unshuffle_2::<2>(src, &mut dest);
        }
        4 => {
            unshuffle_2::<4>(src, &mut dest);
        }
        8 => {
            unshuffle_2::<8>(src, &mut dest);
        }
        _ => unimplemented!(),
    }

    dest
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{BigEndian, ByteOrder};
    use test::Bencher;

    #[test]
    fn shuffle_hdf5_example() {
        let mut v: [i32; 5] = [1, 23, 43, 56, 35];
        BigEndian::from_slice_i32(&mut v);

        assert_eq!(
            v.as_byte_slice(),
            [
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x2B, 0x00, 0x00,
                0x00, 0x38, 0x00, 0x00, 0x00, 0x23
            ]
        );

        let mut d = vec![0_u8; 5 * 4];

        shuffle(&v, &mut d);

        assert_eq!(
            d,
            [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x01, 0x17, 0x2B, 0x38, 0x23,
            ]
        );
    }

    #[bench]
    fn shuffle_4kb(b: &mut Bencher) {
        use rand::distributions::Standard;
        use rand::{thread_rng, Rng};

        let v: Vec<i32> = thread_rng().sample_iter(Standard).take(1024).collect();
        let mut d = vec![0_u8; 1024 * 4];

        b.iter(|| shuffle(&v, &mut d))
    }

    #[bench]
    fn unshuffle_4kb(b: &mut Bencher) {
        use rand::distributions::Standard;
        use rand::{thread_rng, Rng};

        let v: Vec<i32> = thread_rng().sample_iter(Standard).take(1024).collect();
        let mut d = vec![0_u8; 1024 * 4];

        b.iter(|| shuffle(&v, &mut d))
    }

    #[bench]
    fn shuffle_4mb(b: &mut Bencher) {
        use rand::distributions::Standard;
        use rand::{thread_rng, Rng};

        let v: Vec<u8> = thread_rng()
            .sample_iter(Standard)
            .take(4 * 1024 * 1024)
            .collect();
        let mut d = vec![0_i32; 1024 * 1024];

        b.iter(|| unshuffle(&v, &mut d))
    }

    #[bench]
    fn unshuffle_4mb(b: &mut Bencher) {
        use rand::distributions::Standard;
        use rand::{thread_rng, Rng};

        let v: Vec<u8> = thread_rng()
            .sample_iter(Standard)
            .take(4 * 1024 * 1024)
            .collect();
        let mut d = vec![0_i32; 1024 * 1024];

        b.iter(|| unshuffle(&v, &mut d))
    }

    #[bench]
    fn unshuffle_sized_4mb(b: &mut Bencher) {
        use rand::distributions::Standard;
        use rand::{thread_rng, Rng};

        let v: Vec<u8> = thread_rng()
            .sample_iter(Standard)
            .take(4 * 1024 * 1024)
            .collect();

        b.iter(|| unshuffle_sized(&v, 4))
    }

    #[bench]
    fn unshuffle_bytes_4mb(b: &mut Bencher) {
        use rand::distributions::Standard;
        use rand::{thread_rng, Rng};

        let v: Vec<u8> = thread_rng()
            .sample_iter(Standard)
            .take(4 * 1024 * 1024)
            .collect();
        let vb: Bytes = Bytes::from(v);

        b.iter(|| unshuffle_bytes(&vb, 4))
    }

    #[test]
    fn shuffle_unshuffle() {
        let v: [i32; 5] = [1, 23, 43, 56, 35];
        let mut d = vec![0_u8; 5 * 4];

        shuffle(&v, &mut d);

        let mut d2: Vec<i32> = vec![0; 5];

        unshuffle(&d, &mut d2);

        assert_eq!(v, d2.as_slice());
    }
}
