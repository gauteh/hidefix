use super::byteorder::{to_big_e_sized, Order, ToBigEndian, ToNative};
use crate::idx::Datatype;
use byte_slice_cast::{AsMutSliceOf, FromByteSlice};
use itertools::izip;

pub fn xdr_factor(dtype: Datatype) -> usize {
    use Datatype::*;

    match dtype {
        Custom(_) => 1,
        _ => {
            if dtype.dsize() < 4 {
                4 / dtype.dsize()
            } else {
                1
            }
        }
    }
}

/// Upcast and convert to big-endian.
fn xdr_cast_slice<S, D>(mut src: Vec<u8>, order: Order) -> Result<Vec<u8>, anyhow::Error>
where
    S: ToNative + FromByteSlice + Copy,
    D: ToBigEndian + FromByteSlice + From<S> + Copy,
{
    let u: &mut [S] = src.as_mut_slice_of::<S>()?;

    let scale: usize = std::mem::size_of::<D>() / std::mem::size_of::<S>();
    assert!(scale > 1);

    let mut n: Vec<u8> = vec![0; u.len() * std::mem::size_of::<D>()];

    let nn: &mut [D] = n.as_mut_slice_of::<D>()?;

    for (s, d) in izip!(u, nn) {
        s.to_native(order);
        *d = Into::<D>::into(*s);
        d.to_big_e(Order::native())
    }

    Ok(n)
}

pub fn xdr(mut src: Vec<u8>, dtype: Datatype, order: Order) -> Result<Vec<u8>, anyhow::Error> {
    use Datatype::*;

    match dtype {
        Custom(_) => Ok(src),

        UInt(1) => xdr_cast_slice::<u8, u32>(src, order),
        UInt(2) => xdr_cast_slice::<u16, u32>(src, order),
        Int(1) => xdr_cast_slice::<i8, i32>(src, order),
        Int(2) => xdr_cast_slice::<i16, i32>(src, order),

        _ => {
            to_big_e_sized(&mut src, order, dtype.dsize())?;
            Ok(src)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::byteorder::Swap;
    use byte_slice_cast::{AsByteSlice, AsSliceOf};

    #[test]
    fn dsize() {
        assert_eq!(xdr_factor(Datatype::Int(2)), 2);
        assert_eq!(xdr_factor(Datatype::Int(4)), 1);
        assert_eq!(xdr_factor(Datatype::UInt(1)), 4);
        assert_eq!(xdr_factor(Datatype::UInt(2)), 2);
        assert_eq!(xdr_factor(Datatype::UInt(4)), 1);
        assert_eq!(xdr_factor(Datatype::Custom(7)), 1);
        assert_eq!(xdr_factor(Datatype::Custom(1)), 1);
    }

    #[test]
    fn test_u16() {
        let src0 = vec![0u16, 10u16];
        let src1 = src0.as_byte_slice().to_vec();
        let x = xdr(src1, Datatype::UInt(2), Order::native()).unwrap();
        let x = x.as_slice_of::<u32>().unwrap();

        for (s, d) in izip!(src0, x) {
            assert_eq!(s as u32, d.swap_bytes());
        }
    }

    #[test]
    fn test_u8() {
        let src0 = vec![0u8, 10u8];
        let src1 = src0.as_byte_slice().to_vec();
        let x = xdr(src1, Datatype::UInt(1), Order::native()).unwrap();
        let x = x.as_slice_of::<u32>().unwrap();

        for (s, d) in izip!(src0, x) {
            assert_eq!(s as u32, d.swap_bytes());
        }
    }

    #[test]
    fn test_f32() {
        let src0 = vec![0f32, 10f32];
        let src1 = src0.as_byte_slice().to_vec();
        let x = xdr(src1, Datatype::Float(4), Order::native()).unwrap();
        let x = x.as_slice_of::<f32>().unwrap();

        for (s, d) in izip!(src0, x) {
            let d = d.swap();
            assert_eq!(s, d);
        }
    }

    #[test]
    fn test_i16() {
        let src0 = vec![1i16, 128i16];
        let src1 = src0.as_byte_slice().to_vec();
        println!("{:?}", src1);
        let x = xdr(src1, Datatype::Int(2), Order::native()).unwrap();
        println!("{:?}", x);
        let x = x.as_slice_of::<i32>().unwrap();

        for (s, d) in izip!(src0, x) {
            assert_eq!(s as i32, d.swap_bytes());
        }
    }
}
