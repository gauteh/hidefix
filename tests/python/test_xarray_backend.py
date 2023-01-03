import xarray as xr
from hidefix import xarray

def test_coads(coads):
    ds = xr.open_dataset(coads, engine='hidefix')
    print(ds)

