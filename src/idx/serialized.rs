//! Versioned serialization wrapper for an [`Index`], carrying a fingerprint of
//! the source file so consumers can detect a stale index.
//!
//! The byte layout is a fixed header followed by a self-describing flexbuffers
//! payload:
//!
//! ```text
//! b"HFXI" | format_version: u32 LE | flexbuffers(SerializedIndex)
//! ```
//!
//! The version sits in front of the payload so it can be checked before the
//! payload encoding is interpreted. The payload itself is flexbuffers rather
//! than bincode: flexbuffers is self-describing, so fields added later with
//! `#[serde(default)]` keep older serialized indexes readable.

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::index::Index;

/// Magic bytes identifying a serialized hidefix index.
pub const MAGIC: [u8; 4] = *b"HFXI";

/// Current serialization format version.
pub const FORMAT_VERSION: u32 = 1;

/// A serialized [`Index`] together with a fingerprint (path, size, mtime) of
/// the source file the index was built from. A size or mtime of `0` means
/// unknown; staleness policy is left to the consumer.
///
/// The fingerprint is best-effort, not cryptographic: `source_mtime` has
/// whole-second granularity, so a rewrite that lands within the same second
/// and leaves the file the same size is indistinguishable from no change at
/// all. Consumers that need certainty about content should hash the source
/// themselves rather than rely on this.
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedIndex {
    /// Path of the source file at indexing time.
    #[serde(default)]
    pub source_path: Option<String>,
    /// Size in bytes of the source file (0 if unknown).
    #[serde(default)]
    pub source_size: u64,
    /// Modification time of the source file (unix seconds, 0 if unknown).
    #[serde(default)]
    pub source_mtime: i64,
    /// The flexbuffers-serialized [`Index`].
    #[serde(with = "serde_bytes")]
    index: Vec<u8>,
}

impl SerializedIndex {
    /// Wrap `idx` with the given source-file fingerprint (`0` for unknown),
    /// see [`file_fingerprint`].
    pub fn from_index(
        idx: &Index,
        source_size: u64,
        source_mtime: i64,
    ) -> Result<SerializedIndex, anyhow::Error> {
        let mut s = flexbuffers::FlexbufferSerializer::new();
        idx.serialize(&mut s)?;

        Ok(SerializedIndex {
            source_path: idx.path().map(|p| p.to_string_lossy().into_owned()),
            source_size,
            source_mtime,
            index: s.take_buffer(),
        })
    }

    /// Serialize to bytes, prefixed with the magic and format version header.
    pub fn to_bytes(&self) -> Result<Vec<u8>, anyhow::Error> {
        let mut s = flexbuffers::FlexbufferSerializer::new();
        self.serialize(&mut s)?;

        let mut b = Vec::with_capacity(MAGIC.len() + 4 + s.view().len());
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        b.extend_from_slice(s.view());
        Ok(b)
    }

    /// Parse bytes produced by [`SerializedIndex::to_bytes`]. Fails with a
    /// clear message on truncated or foreign bytes or an unknown format
    /// version.
    pub fn from_bytes(b: &[u8]) -> Result<SerializedIndex, anyhow::Error> {
        anyhow::ensure!(b.len() >= 8, "truncated serialized index");
        anyhow::ensure!(
            b[..4] == MAGIC,
            "not a serialized hidefix index (bad magic)"
        );
        let version = u32::from_le_bytes(b[4..8].try_into().unwrap());
        anyhow::ensure!(
            version == FORMAT_VERSION,
            "unsupported hidefix index format version {version} (expected {FORMAT_VERSION}): \
             re-serialize the index with a matching hidefix"
        );
        let r = flexbuffers::Reader::get_root(&b[8..])
            .map_err(|e| anyhow::anyhow!("invalid serialized index payload: {e}"))?;
        Ok(SerializedIndex::deserialize(r)?)
    }

    /// Deserialize the wrapped [`Index`].
    pub fn index(&self) -> Result<Index<'static>, anyhow::Error> {
        let r = flexbuffers::Reader::get_root(self.index.as_slice())?;
        Ok(Index::deserialize(r)?.into_owned())
    }
}

/// Size and modification time (unix seconds) of `path`, `(0, 0)` where
/// unavailable.
pub fn file_fingerprint(path: &Path) -> (u64, i64) {
    std::fs::metadata(path)
        .map(|m| {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            (m.len(), mtime)
        })
        .unwrap_or((0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    const COADS: &str = "tests/data/coads_climatology.nc4";

    #[test]
    fn wrapper_roundtrip() {
        let i = Index::index(COADS).unwrap();
        let (size, mtime) = file_fingerprint(Path::new(COADS));
        assert!(size > 0);
        assert!(mtime > 0);

        let s = SerializedIndex::from_index(&i, size, mtime).unwrap();
        let b = s.to_bytes().unwrap();

        let d = SerializedIndex::from_bytes(&b).unwrap();
        assert_eq!(d.source_path.as_deref(), Some(COADS));
        assert_eq!(d.source_size, size);
        assert_eq!(d.source_mtime, mtime);

        let di = d.index().unwrap();
        assert_eq!(di.datasets().len(), i.datasets().len());
        assert_eq!(
            di.dataset_dim_names("SST").unwrap(),
            ["TIME", "COADSY", "COADSX"]
        );

        // the owned index must read independently of the serialized buffer.
        drop(d);
        let mut r = di.reader("SST").unwrap();
        r.values::<f32, _>(..).unwrap();
    }

    #[test]
    fn unknown_version_rejected() {
        let i = Index::index(COADS).unwrap();
        let mut b = SerializedIndex::from_index(&i, 0, 0)
            .unwrap()
            .to_bytes()
            .unwrap();
        b[4..8].copy_from_slice(&2u32.to_le_bytes());

        let e = SerializedIndex::from_bytes(&b).unwrap_err();
        assert!(e.to_string().contains("format version 2"), "{e}");
    }

    #[test]
    fn bad_magic_rejected() {
        let e = SerializedIndex::from_bytes(b"not an index at all").unwrap_err();
        assert!(e.to_string().contains("bad magic"), "{e}");
    }

    #[test]
    fn truncated_rejected() {
        for b in [&b""[..], &b"HFX"[..], &b"HFXI\x01\x00\x00"[..]] {
            let e = SerializedIndex::from_bytes(b).unwrap_err();
            assert!(e.to_string().contains("truncated"), "{e}");
        }
    }

    #[test]
    fn empty_payload_rejected() {
        // header only, valid magic and version: an invalid payload, not bad magic.
        let mut b = MAGIC.to_vec();
        b.extend_from_slice(&FORMAT_VERSION.to_le_bytes());

        let e = SerializedIndex::from_bytes(&b).unwrap_err();
        assert!(
            e.to_string().contains("invalid serialized index payload"),
            "{e}"
        );
    }
}
