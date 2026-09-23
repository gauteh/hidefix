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


def assert_engines_equal(path, hidefix_kwargs=None, **kwargs):
    """The hidefix engine must produce the same dataset as the netcdf4 engine.

    `hidefix_kwargs` are passed to the hidefix engine only (e.g. `index`)."""
    hfx = xr.open_dataset(path, engine='hidefix', **kwargs,
                          **(hidefix_kwargs or {}))
    ncd = xr.open_dataset(path, engine='netcdf4', **kwargs)

    assert set(hfx.data_vars) == set(ncd.data_vars)
    assert set(hfx.coords) == set(ncd.coords)
    assert dict(hfx.sizes) == dict(ncd.sizes)

    assert set(hfx.attrs) == set(ncd.attrs)
    for k, v in ncd.attrs.items():
        np.testing.assert_array_equal(hfx.attrs[k], v)

    for name in ncd.variables:
        assert hfx[name].dims == ncd[name].dims
        assert set(hfx[name].attrs) == set(ncd[name].attrs)
        for k, v in ncd[name].attrs.items():
            np.testing.assert_array_equal(hfx[name].attrs[k], v)
        # decoded dtype must match: a float32 scale_factor/add_offset must
        # unpack to float32, not silently upcast to float64.
        assert hfx[name].dtype == ncd[name].dtype, (
            f"{name}: {hfx[name].dtype} != {ncd[name].dtype}")
        np.testing.assert_array_equal(hfx[name].values, ncd[name].values)


def test_engine_equality_coads(coads):
    # the coads time units ('hour since 0000-01-01') are not decodable by any
    # engine with current xarray/cftime, so compare with time decoding off.
    assert_engines_equal(coads, decode_times=False)


def test_engine_equality_decoded_time(tmp_path):
    import netCDF4 as nc4

    path = tmp_path / 'time.nc'
    ds = nc4.Dataset(path, 'w')
    ds.setncattr('description', 'engine equality')
    ds.createDimension('time', 4)
    time = ds.createVariable('time', np.float64, ('time', ))
    time.units = 'hours since 2000-01-01 00:00:00'
    time.calendar = 'standard'
    time[:] = np.arange(4)
    t = ds.createVariable('t', np.float32, ('time', ))
    t.units = 'K'
    t[:] = np.arange(4, dtype=np.float32)
    ds.close()

    assert_engines_equal(path)

    hfx = xr.open_dataset(path, engine='hidefix')
    assert np.issubdtype(hfx['time'].dtype, np.datetime64)


def test_engine_equality_packed(tmp_path):
    # A packed int16 variable with float32 scale_factor/add_offset must unpack
    # to float32 through both engines; a float64-scaled variable is included for
    # contrast. netCDF4 is the write-side oracle.
    import netCDF4 as nc4

    path = tmp_path / 'packed.nc'
    ds = nc4.Dataset(path, 'w')
    ds.createDimension('x', 5)

    p = ds.createVariable('packed', np.int16, ('x', ))
    p.scale_factor = np.float32(0.1)
    p.add_offset = np.float32(5.0)
    p[:] = np.arange(5, dtype=np.int16)

    q = ds.createVariable('packed64', np.int16, ('x', ))
    q.scale_factor = np.float64(0.1)
    q.add_offset = np.float64(5.0)
    q[:] = np.arange(5, dtype=np.int16)

    ds.close()

    assert_engines_equal(path)

    # explicit dtype expectations (guards against both engines agreeing on the
    # wrong upcast, which cross-engine equality alone would not catch).
    hfx = xr.open_dataset(path, engine='hidefix')
    assert hfx['packed'].dtype == np.float32
    assert hfx['packed64'].dtype == np.float64

    # the raw (undecoded) scale_factor/add_offset must themselves be float32.
    raw = xr.open_dataset(path, engine='hidefix', mask_and_scale=False)
    assert raw['packed'].attrs['scale_factor'].dtype == np.float32
    assert raw['packed'].attrs['add_offset'].dtype == np.float32


@pytest.mark.skip(reason = 'xarray, cftime, pandas no longer manages to decode dates here')
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
