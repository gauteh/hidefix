pub enum Order {
    BE,
    LE
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

impl Swap for u32 {
    fn swap(&self) -> Self {
        self.swap_bytes()
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

impl<T> ToNative for [T] where T: Swap {
    fn to_native(&mut self, order: Order) {
        if cfg!(target_endian = "big") {
            match order {
                Order::BE => (),
                Order::LE => for n in self { *n = n.swap() }
            }
        } else {
            match order {
                Order::BE => for n in self { *n = n.swap() },
                Order::LE => (),
            }
        }
    }
}

impl<T> ToBigEndian for [T] where T: Swap {
    fn to_big_e(&mut self, order: Order) {
        match order {
            Order::BE => (),
            Order::LE => for n in self { *n = n.swap() }
        }
    }
}

