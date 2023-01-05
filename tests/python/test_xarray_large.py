import pytest
import xarray as xr
from hidefix import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def test_large_hf(large_file, plot):
    f, v = large_file
    ds = xr.open_dataset(f, engine='hidefix')
    print(ds)

    if plot:
        ds.temperature.isel(time=0, depth=0).plot()
        plt.show()

def test_large_nc(large_file, plot):
    f, v = large_file
    ds = xr.open_dataset(f, engine='netcdf4')
    print(ds)
    if plot:
        ds.temperature.isel(time=0, depth=0).plot()
        plt.show()

@pytest.mark.parametrize("engine", ["netcdf4", "hidefix"])
def test_read_large(large_file, benchmark, engine):
    f, v = large_file

    def setup():
        return (xr.open_dataset(f, engine=engine), v), {}

    def read(ds, v):
        return ds[v].load()
        # return ds[v].values

    vals = benchmark.pedantic(read, setup=setup)
    # print(vals.shape, type(vals))
    print(vals)
