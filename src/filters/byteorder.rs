use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Order {
    BE,
    LE,
    Unknown,
}

impl Order {
    pub fn native() -> Self {
        if cfg!(target_endian = "big") {
            Order::BE
        } else {
            Order::LE
        }
    }
}

impl From<hdf5::datatype::ByteOrder> for Order {
    fn from(byo: hdf5::datatype::ByteOrder) -> Self {
        use hdf5::datatype::ByteOrder;

        match byo {
            ByteOrder::BigEndian => Order::BE,
            ByteOrder::LittleEndian => Order::LE,
            _ => Order::Unknown,
        }
    }
}

pub trait ToNative {
    /// `order` is the original order of the bytes.
    fn to_native(&mut self, order: Order);
}

pub trait ToBigEndian {
    /// `order` is the original order of the bytes.
    fn to_big_e(&mut self, order: Order);
}

pub trait Swap {
    fn swap(&self) -> Self;
}

impl Swap for u16 {
    fn swap(&self) -> Self {
        self.swap_bytes()
    }
}

impl Swap for u32 {
    fn swap(&self) -> Self {
        self.swap_bytes()
    }
}

impl Swap for u64 {
    fn swap(&self) -> Self {
        self.swap_bytes()
    }
}

impl Swap for i32 {
    fn swap(&self) -> Self {
        self.swap_bytes()
    }
}

impl Swap for f32 {
    fn swap(&self) -> Self {
        unsafe {
            let int = *(self as *const f32 as *const u32);
            *(&int.to_be() as *const u32 as *const f32)
        }
    }
}

impl Swap for f64 {
    fn swap(&self) -> Self {
        unsafe {
            let int = *(self as *const f64 as *const u64);
            *(&int.to_be() as *const u64 as *const f64)
        }
    }
}

impl ToNative for [u8] {
    // no-op
    fn to_native(&mut self, _order: Order) {}
}

impl ToBigEndian for [u8] {
    // no-op
    fn to_big_e(&mut self, _order: Order) {}
}

impl<T> ToNative for [T]
where
    T: Swap,
{
    fn to_native(&mut self, order: Order) {
        if cfg!(target_endian = "big") {
            match order {
                Order::BE => (),
                Order::LE => {
                    for n in self {
                        *n = n.swap()
                    }
                }
                _ => unimplemented!(),
            }
        } else {
            match order {
                Order::BE => {
                    for n in self {
                        *n = n.swap()
                    }
                }
                Order::LE => (),
                _ => unimplemented!(),
            }
        }
    }
}

impl<T> ToBigEndian for [T]
where
    T: Swap,
{
    fn to_big_e(&mut self, order: Order) {
        match order {
            Order::BE => (),
            Order::LE => {
                for n in self {
                    *n = n.swap()
                }
            }
            _ => unimplemented!(),
        }
    }
}

/// Swap bytes of `buf` if necessary so that they are big_endian. `dsz` is the size of the data
/// type.
///
/// Fails if `buf` size is not a multiple of `dsz`.
pub fn to_big_e_sized(buf: &mut [u8], order: Order, dsz: usize) -> Result<(), anyhow::Error> {
    use byte_slice_cast::AsMutSliceOf;

    if let Order::LE = order {
        match dsz {
            1 => (),
            2 => {
                let v = buf.as_mut_slice_of::<u16>()?;
                v.to_big_e(order);
            }
            4 => {
                let v = buf.as_mut_slice_of::<u32>()?;
                v.to_big_e(order);
            }
            8 => {
                let v = buf.as_mut_slice_of::<u64>()?;
                v.to_big_e(order);
            }
            _ => unimplemented!(),
        }
    }

    Ok(())
}
