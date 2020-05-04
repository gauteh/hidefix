# HIDEFIX

<img src="https://raw.githubusercontent.com/gauteh/hidefix/master/idefix.png">

This builds an index of a [HDF5](https://support.hdfgroup.org/HDF5/doc/H5.format.html) file for direct and concurrent access to data chunks.

## Inspiration and other projects

This work based in part on the [DMR++ module](https://github.com/OPENDAP/bes/tree/master/modules/dmrpp_module) of the [OPeNDAP](https://www.opendap.org/) [Hyrax server](https://www.opendap.org/software/hyrax-data-server).

The [zarr](https://zarr.readthedocs.io/en/stable/) format does something similar, and the same approach has been [tested out on HDF5](https://medium.com/pangeo/cloud-performant-reading-of-netcdf4-hdf5-data-using-the-zarr-library-1a95c5c92314) as swell.

