import xarray as xr
import numpy as np
import pytest
import os

try:
    import matplotlib.pyplot as plt
except:
    pass


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


def test_xarray_mfdataset(data):
    urls = [str(data / 'jan.nc4'), str(data / 'feb.nc4')]
    ds = xr.decode_cf(xr.open_mfdataset(urls, engine='hidefix'))
    print(ds)


@pytest.mark.skipif(not os.path.exists(
    '/lustre/storeB/project/fou/om/NORA3/equinor/atm_hourly/arome3km_1hr_198501.nc'
),
                    reason='nora3 data not available')
def test_xarray_mfdataset_nora3():
    urls = [
        '/lustre/storeB/project/fou/om/NORA3/equinor/atm_hourly/arome3km_1hr_198501.nc',
        '/lustre/storeB/project/fou/om/NORA3/equinor/atm_hourly/arome3km_1hr_198502.nc'
    ]
    ds = xr.open_mfdataset(urls, engine='hidefix')
    print(ds)
