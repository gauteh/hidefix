use libdeflater::Decompressor;

/// Helper to decompress a gzipped slice of `u8`s to another buffer of `u8`s.
pub fn decompress(compressed: &[u8], out: &mut [u8]) -> Result<usize, anyhow::Error> {
    let mut de = Decompressor::new();
    de.zlib_decompress(compressed, out)
        .map_err(|_| anyhow!("Could not decompress chunk"))
}
