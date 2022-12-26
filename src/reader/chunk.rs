use crate::filters;
use std::io::{Read, Seek, SeekFrom};

pub(crate) fn read_chunk_to<F>(
    fd: &mut F,
    addr: u64,
    dst: &mut [u8],
) -> Result<(), anyhow::Error>
where
    F: Read + Seek,
{
    fd.seek(SeekFrom::Start(addr))?;
    fd.read_exact(dst)?;
    Ok(())
}

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

    let mut cache: Vec<u8> = vec![0; size as usize];

    fd.seek(SeekFrom::Start(addr))?;
    fd.read_exact(&mut cache)?;

    // Decompress
    let cache = if gzipped {
        let mut decache = vec![0; chunk_sz as usize];

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

pub(crate) fn decode_chunk(
    chunk: Vec<u8>,
    chunk_sz: u64,
    dsz: u64,
    gzipped: bool,
    shuffled: bool,
) -> Result<Vec<u8>, anyhow::Error> {
    debug_assert!(dsz < 16); // unlikely data-size

    // Decompress
    let cache = if gzipped {
        let mut decache = vec![0; chunk_sz as usize];

        filters::gzip::decompress(&chunk, &mut decache)?;

        debug_assert_eq!(decache.len(), chunk_sz as usize);

        decache
    } else {
        chunk
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
