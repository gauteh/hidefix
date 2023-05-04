[![Crates.io](https://img.shields.io/crates/v/hidefix.svg)](https://crates.io/crates/hidefix)
[![PyPI](https://img.shields.io/pypi/v/hidefix.svg)](https://pypi.org/project/hidefix/)
[![Documentation](https://docs.rs/hidefix/badge.svg)](https://docs.rs/hidefix/)
[![Build (rust)](https://github.com/gauteh/hidefix/workflows/Rust/badge.svg)](https://github.com/gauteh/hidefix/actions?query=branch%3Amain)
[![Build (python)](https://github.com/gauteh/hidefix/workflows/Python/badge.svg)](https://github.com/gauteh/hidefix/actions?query=branch%3Amain)
[![codecov](https://codecov.io/gh/gauteh/hidefix/branch/main/graph/badge.svg)](https://codecov.io/gh/gauteh/hidefix)
[![Rust nightly](https://img.shields.io/badge/rustc-nightly-orange)](https://rust-lang.github.io/rustup/installation/other.html)

<img src="https://raw.githubusercontent.com/gauteh/hidefix/main/idefix.png">

# HIDEFIX

This Rust and Python library provides an alternative reader for the
[HDF5](https://support.hdfgroup.org/HDF5/doc/H5.format.html) file or [NetCDF4
file](https://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html)
(which uses HDF5) which supports concurrent access to data. This is achieved by
building an index of the chunks, allowing a thread to use many file handles to
read the file. The original (native) HDF5 library is used to build the index,
but once it has been created it is no longer needed. The index can be
serialized to disk so that performing the indexing is not necessary.

In Rust:

```rust
use hidefix::prelude::*;

let idx = Index::index("tests/data/coads_climatology.nc4").unwrap();
let mut r = idx.reader("SST").unwrap();

let values = r.values::<f32>(None, None).unwrap();

println!("SST: {:?}", values);
```

or with Python using Xarray:
```python
import xarray as xr
import hidefix

ds = xr.open_dataset('file.nc', engine='hidefix')
print(ds)
```

## Motivation

The HDF5 library requires internal locks to be _thread-safe_ since it relies on
internal buffers which cannot be safely accessed/written to from multiple
threads. This effectively causes multi-threaded applications to use sequential
reads, while competing for the locks. And also apparently cause each other
trouble, perhaps through dropping cached chunks which other threads still need.
It can be safely used from different processes, but that requires potentially
much more overhead than multi-threaded or asynchronous code.

## Some basic benchmarks

`hidefix` is intended to perform better when concurrent reads are made either
to the same dataset, same file or to different files from a single process. For
basic benchmarks the performance is on-par or slightly better compared to doing
standard *sequential* reads than the native HDF5 library (through its
[rust-bindings](https://github.com/aldanor/hdf5-rust)). Where `hidefix` shines
is once the _multiple threads_ in the _same process_ tries to read in _any way_
from a HDF5 file simultaneously.

This simple benchmark tries to read a small dataset sequentially or
concurrently using the `cached` reader from `hidefix` and the native reader
from HDF5. The dataset is chunked, shuffled and compressed (using gzip):

```sh
$ cargo bench --bench concurrency -- --ignored

test shuffled_compressed::cache_concurrent_reads  ... bench:  15,903,406 ns/iter (+/- 220,824)
test shuffled_compressed::cache_sequential        ... bench:  59,778,761 ns/iter (+/- 602,316)
test shuffled_compressed::native_concurrent_reads ... bench: 411,605,868 ns/iter (+/- 35,346,233)
test shuffled_compressed::native_sequential       ... bench: 103,457,237 ns/iter (+/- 7,703,936)
```

## Inspiration and other projects

This work is based in part on the [DMR++
module](https://github.com/OPENDAP/bes/tree/master/modules/dmrpp_module) of the
[OPeNDAP](https://www.opendap.org/) [Hyrax
server](https://www.opendap.org/software/hyrax-data-server). The
[zarr](https://zarr.readthedocs.io/en/stable/) format does something similar,
and the same approach has been [tested out on
HDF5](https://medium.com/pangeo/cloud-performant-reading-of-netcdf4-hdf5-data-using-the-zarr-library-1a95c5c92314)
as swell.

