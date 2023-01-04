import xarray as xr
from hidefix import xarray
import numpy as np
import matplotlib.pyplot as plt

def test_coads_hf(coads, plot):
    ds = xr.open_dataset(coads, engine='hidefix', decode_times=False)
    print(ds)

    sst = ds['SST']
    print(sst)
    print(sst.shape)
    print(sst.values.shape)


    if plot:
        sst.plot()
        plt.show()

    dsnc = xr.open_dataset(coads, engine='netcdf4', decode_times=False)
    np.testing.assert_array_equal(ds['SST'], dsnc['SST'])


def test_coads_nc(coads, plot):
    ds = xr.open_dataset(coads, engine='netcdf4', decode_times=False)
    print(ds)

    if plot:
        ds['SST'].plot()
        plt.show()
