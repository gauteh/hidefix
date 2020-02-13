use byte_slice_cast::{AsByteSlice, AsMutByteSlice, ToByteSlice, ToMutByteSlice};

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

    // TODO: check if rustc unrolls loop or we should manually implement Duff's device.
    for i in 0..wsz {
        // chunks
        for j in 0..sz {
            // words
            dest[i * sz + j] = src[j * wsz + i];
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
            dest[j * wsz + i] = src[i * sz + j];
        }
    }
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
    fn shuffle_4mb(b: &mut Bencher) {
        use rand::distributions::Standard;
        use rand::{thread_rng, Rng};

        let v: Vec<i32> = thread_rng()
            .sample_iter(Standard)
            .take(1024 * 1024)
            .collect();
        let mut d = vec![0_u8; 1024 * 1024 * 4];

        b.iter(|| shuffle(&v, &mut d))
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
