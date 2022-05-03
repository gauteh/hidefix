use crate::filters;
use std::io::{Read, Seek, SeekFrom};

pub(crate) fn read_chunk<F>(
    fd: &mut F,
    addr: u64,
    size: u64,
    chunk_sz: u64,
    dsz: u64,
    gzipped: bool,
    shuffled: bool,
    tokio_block: bool,
) -> Result<Vec<u8>, anyhow::Error>
where
    F: Read + Seek,
{
    debug_assert!(dsz < 16); // unlikely data-size

    let mut cache: Vec<u8> = Vec::with_capacity(size as usize);
    unsafe {
        cache.set_len(size as usize);
    }

    fd.seek(SeekFrom::Start(addr))?;
    fd.read_exact(&mut cache)?;

    // Decompress
    let cache = if gzipped {
        let mut decache: Vec<u8> = Vec::with_capacity(chunk_sz as usize);
        unsafe {
            decache.set_len(chunk_sz as usize);
        }

        if tokio_block {
            // TODO: Feature guard to enable single-threaded non-tokio RT's
            tokio::task::block_in_place(|| filters::gzip::decompress(&cache, &mut decache))?;
        } else {
            filters::gzip::decompress(&cache, &mut decache)?;
        }

        debug_assert_eq!(decache.len(), chunk_sz as usize);

        decache
    } else {
        cache
    };

    // Unshuffle
    // TODO: Keep buffers around to avoid allocations.
    // TODO: Write directly to buf_slice when on last filter.
    let cache = if shuffled && dsz > 1 {
        filters::shuffle::unshuffle_sized(&cache, dsz as usize)
    } else {
        cache
    };

    // TODO:
    // * more filters..

    Ok(cache)
}
