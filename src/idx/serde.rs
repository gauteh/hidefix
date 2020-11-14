use serde::de::Error;
use serde::ser::{Serialize, Serializer};
use serde::{Deserialize, Deserializer};

use std::borrow::Cow;
use strength_reduce::StrengthReducedU64;

use super::chunk::{Chunk, ULE};

pub mod sr_u64 {
    use super::*;

    pub fn serialize<S, const D: usize>(
        v: &[StrengthReducedU64; D],
        s: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let b: Vec<u64> = v.iter().map(|s| s.get()).collect();
        b.serialize(s)
    }

    pub fn deserialize<'de, De, const D: usize>(d: De) -> Result<[StrengthReducedU64; D], De::Error>
    where
        De: Deserializer<'de>,
    {
        let mut sr: [StrengthReducedU64; D] = [StrengthReducedU64::new(1); D];

        let v = Vec::<u64>::deserialize(d)?;

        if v.len() != sr.len() {
            return Err(De::Error::custom("length mismatch"));
        }

        for (e, s) in v.iter().zip(sr.iter_mut()) {
            *s = StrengthReducedU64::new(*e);
        }

        Ok(sr)
    }
}

pub mod arr_u64 {
    use super::*;

    pub fn serialize<S, const D: usize>(v: &[u64; D], s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        v.as_slice().serialize(s)
    }

    pub fn deserialize<'de: 'a, 'a, D, const DN: usize>(d: D) -> Result<[u64; DN], D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = <Vec<u64>>::deserialize(d)?;
        let mut a: [u64; DN] = [0; DN];

        for (e, a) in v.iter().zip(a.iter_mut()) {
            *a = *e;
        }

        Ok(a)
    }
}

pub mod chunks_u64s {
    use super::*;

    pub fn serialize<S, const D: usize>(chunks: &[Chunk<D>], s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use zerocopy::AsBytes;

        let slice: &[u8] = Chunk::<D>::slice_as_u64s(chunks).as_bytes();
        serde_bytes::serialize(slice, s)
    }

    pub fn deserialize<'de: 'a, 'a, D, const DE: usize>(
        d: D,
    ) -> Result<Cow<'a, [Chunk<DE>]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use zerocopy::LayoutVerified;

        let bytes = <&'a [u8]>::deserialize(d)?;
        let slice = LayoutVerified::new_slice_unaligned(bytes).unwrap();
        let slice: &[ULE] = slice.into_slice();

        let chunks: &'a [Chunk<DE>] = Chunk::<DE>::slice_from_u64s(slice);
        let chunks = Cow::<'a, [Chunk<DE>]>::from(chunks);

        debug_assert!(chunks.is_borrowed());

        Ok(chunks)
    }
}
