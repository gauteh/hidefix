[![Build Status](https://travis-ci.org/gauteh/hidefix.svg?branch=master)](https://travis-ci.org/gauteh/hidefix)

<img src="https://raw.githubusercontent.com/gauteh/hidefix/master/idefix.png">

# HIDEFIX

This library provides an alternative reader for the
[HDF5](https://support.hdfgroup.org/HDF5/doc/H5.format.html) file or [NetCDF4
file](https://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html)
(which uses HDF5) which supports concurrent access to data. This is achieved by
building an index of the chunks, allowing a thread to use many file handles to
read the file. The original (native) HDF5 library is used to build the index,
but once it has been created it is no longer needed. The index can be
serialized to disk so that performing the indexing is not necessary.

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
$ cargo bench --bench concurrency

test shuffled_compressed::cache_concurrent_reads  ... bench:  26,602,751 ns/iter (+/- 24,391,376)
test shuffled_compressed::cache_sequential        ... bench: 101,006,439 ns/iter (+/- 38,184,751)
test shuffled_compressed::native_concurrent_reads ... bench: 455,274,764 ns/iter (+/- 85,119,929)
test shuffled_compressed::native_sequential       ... bench: 105,945,871 ns/iter (+/- 12,992,272)
```

## Inspiration and other projects

This work based in part on the [DMR++
module](https://github.com/OPENDAP/bes/tree/master/modules/dmrpp_module) of the
[OPeNDAP](https://www.opendap.org/) [Hyrax
server](https://www.opendap.org/software/hyrax-data-server). The
[zarr](https://zarr.readthedocs.io/en/stable/) format does something similar,
and the same approach has been [tested out on
HDF5](https://medium.com/pangeo/cloud-performant-reading-of-netcdf4-hdf5-data-using-the-zarr-library-1a95c5c92314)
as swell.

