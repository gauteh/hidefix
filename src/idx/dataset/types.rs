use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
pub enum Datatype {
    UInt(usize),
    Int(usize),
    Float(usize),
    Custom(usize),
}

impl Datatype {
    pub fn dsize(&self) -> usize {
        use Datatype::*;

        match self {
            UInt(sz) | Int(sz) | Float(sz) | Custom(sz) => *sz,
        }
    }
}

impl From<hdf5::Datatype> for Datatype {
    fn from(dtype: hdf5::Datatype) -> Self {
        match dtype {
            _ if dtype.is::<u8>() => Datatype::UInt(dtype.size()),
            _ if dtype.is::<u16>() => Datatype::UInt(dtype.size()),
            _ if dtype.is::<u32>() => Datatype::UInt(dtype.size()),
            _ if dtype.is::<u64>() => Datatype::UInt(dtype.size()),
            _ if dtype.is::<i8>() => Datatype::Int(dtype.size()),
            _ if dtype.is::<i16>() => Datatype::Int(dtype.size()),
            _ if dtype.is::<i32>() => Datatype::Int(dtype.size()),
            _ if dtype.is::<i64>() => Datatype::Int(dtype.size()),
            _ if dtype.is::<f32>() => Datatype::Float(dtype.size()),
            _ if dtype.is::<f64>() => Datatype::Float(dtype.size()),
            _ => Datatype::Custom(dtype.size()),
        }
    }
}
