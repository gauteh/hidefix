//! HDF5 can store chunks in various types of data structures internally (`BTreeMap`, etc.), so
//! it is not necessarily a simple sorted array (presumably because chunks can be added at a later
//! time). The `get_chunk_info` methods iterate over this structure internally to get the requested
//! chunk (based on a predicate function set up internally).
//!
//! It would be far more efficient for us if we could retrieve all chunks through one iteration (N),
//! rather than do a full iteration for all chunks which requires SUM(I) for I in N operations.
//!
//! This module provides bindings to a [proposed `chunks_iter`](https://github.com/HDFGroup/hdf5/pull/6).
#![allow(non_camel_case_types, non_snake_case)]
use libc::{c_int, c_void};

use hdf5_sys::h5::{haddr_t, herr_t, hsize_t};
use hdf5_sys::h5i::hid_t;

pub type H5D_chunk_iter_op_t = Option<
    extern "C" fn(
        offset: *const hsize_t,
        filter_mask: u32,
        addr: haddr_t,
        nbytes: u32,
        op_data: *mut c_void,
    ) -> c_int,
>;

extern "C" {
    #[cfg(test)]
    pub fn H5open() -> herr_t;

    pub fn H5Dchunk_iter(dset: hid_t, cb: H5D_chunk_iter_op_t, op_data: *mut c_void) -> herr_t;
}

/// Holds the rust callback and the number of dimensions (required to build slice).
#[repr(C)]
struct RustCb<F>
where
    F: FnMut(&[u64], u32, u64, u32) -> i32,
{
    pub cb: F,
    pub ndims: usize,
}

extern "C" fn chunks_cb<F>(
    offset: *const hsize_t,
    filter_mask: u32,
    addr: haddr_t,
    nbytes: u32,
    op_data: *mut c_void,
) -> c_int
where
    F: FnMut(&[u64], u32, u64, u32) -> i32,
{
    let cb: *mut RustCb<F> = op_data as _;
    let offset: &[u64] = unsafe { std::slice::from_raw_parts(offset, (*cb).ndims) };

    unsafe { ((*cb).cb)(offset, filter_mask, addr, nbytes) }
}

/// Apply closure to all chunks in dataset. Returning a positive value in the closure will halt the
/// iterator, a negative will cause a failure, while zero continues.
pub fn chunks_foreach<F>(dset: hid_t, cb: F)
where
    F: FnMut(&[u64], u32, u64, u32) -> i32,
{
    use hdf5_sys::h5d::H5Dget_space;
    use hdf5_sys::h5s::{H5Sclose, H5Sget_simple_extent_ndims};

    let ndims = hdf5::sync::sync(|| unsafe {
        let space = H5Dget_space(dset);
        let ndims = H5Sget_simple_extent_ndims(space);
        H5Sclose(space);

        ndims
    });

    let mut rcb = RustCb::<F> {
        cb,
        ndims: ndims as usize,
    };

    let rcptr: *mut RustCb<F> = &mut rcb as *mut _;
    let voidptr: *mut c_void = unsafe { std::mem::transmute(rcptr) };

    let e = hdf5::sync::sync(|| unsafe { H5Dchunk_iter(dset, Some(chunks_cb::<F>), voidptr) });
    dbg!(e);

    assert!(e >= 0);
}

#[derive(Debug, Clone)]
pub struct ChunkInfo {
    pub offset: Vec<u64>,
    pub filter_mask: u32,
    pub addr: u64,
    pub nbytes: u32,
}

/// Collect all chunks in dataset.
#[allow(dead_code)]
pub fn chunks_collect_all(dset: hid_t) -> Vec<ChunkInfo> {
    let mut v = Vec::new();

    chunks_foreach(dset, |offset, filter_mask, addr, nbytes| {
        v.push(ChunkInfo {
            offset: offset.iter().cloned().collect(),
            filter_mask,
            addr,
            nbytes,
        });

        0
    });

    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5_sys::h5d::*;
    use hdf5_sys::h5f::*;
    use hdf5_sys::h5p::*;
    use std::ffi::CString;
    use test::Bencher;

    #[test]
    fn it_works() {
        hdf5::sync::sync(|| {
            let e = unsafe { H5open() };
            assert_eq!(e, 0);
        });
    }

    #[test]
    fn coads_sst() {
        let dset = hdf5::sync::sync(|| unsafe {
            H5open();
            let p = CString::new("tests/data/coads_climatology.nc4").unwrap();
            let hf = H5Fopen(p.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert!(hf > 0);

            let p = CString::new("SST").unwrap();
            let dset = H5Dopen2(hf, p.as_ptr(), H5P_DEFAULT);
            assert!(dset > 0);

            dset
        });

        let mut v = Vec::new();

        chunks_foreach(dset, |offset, filter_mask, addr, nbytes| {
            println!(
                "offset: {:?}, filter_mask: {}, addr: {}, nbytes: {}",
                offset, filter_mask, addr, nbytes
            );
            v.push(addr);
            0
        });
    }

    #[bench]
    fn coads_sst_collect_all(b: &mut Bencher) {
        let hf = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
        let ds = hf.dataset("SST").unwrap();

        b.iter(|| chunks_collect_all(ds.id()));
    }

    #[bench]
    fn coads_sst_collect_via_chunk_info(b: &mut Bencher) {
        let hf = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
        let ds = hf.dataset("SST").unwrap();
        let n = ds.num_chunks().unwrap();

        b.iter(|| {
            (0..n)
                .map(|i| {
                    ds.chunk_info(i)
                        .map(|ci| ChunkInfo {
                            offset: ci.offset,
                            nbytes: ci.size as u32,
                            addr: ci.addr,
                            filter_mask: ci.filter_mask,
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>()
        });
    }
}
