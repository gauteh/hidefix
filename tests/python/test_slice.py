from hidefix import Index
from netCDF4 import Dataset
import numpy as np


def test_coads_len_shape(coads):
    i = Index(coads)
    ds = i.dataset('SST')  # this takes a couple of hundred nanoseconds.

    print(len(ds))
    print("shape =", ds.shape())
    print("chunk shape =", ds.chunk_shape())


def test_feb_slice(feb):
    i = Index(feb)
    ds = i.dataset('T')  # this takes a couple of hundred nanoseconds.
    arr = ds[0:10, 0:2, 0:4].reshape(10, 2, 4)
    # arr[arr<100000] = np.nan
    print(arr, type(arr), arr.shape)

    n = Dataset(feb)
    nds = n['T']
    print("nds.shape =", nds.shape)
    narr = nds[0:10, 0:2, 0:4].filled(np.nan)
    print(narr)

    assert narr.shape == arr.shape

    print(np.all(np.isnan(narr)))

    np.testing.assert_array_equal(arr, narr)


def test_coads_slice(coads):
    i = Index(coads)
    ds = i.dataset('SST')  # this takes a couple of hundred nanoseconds.
    arr = ds[0:2, :, :].reshape(2, 90, 180)
    arr[arr < -100000] = np.nan
    print(arr, type(arr), arr.shape)

    n = Dataset(coads)
    nds = n['SST']
    print("nds.shape =", nds.shape)
    narr = nds[0:2, :, :].filled(np.nan)
    print(narr)

    # assert len(narr.ravel()) == len(arr)
    assert narr.shape == arr.shape

    print(np.all(np.isnan(narr)))

    np.testing.assert_array_equal(arr, narr)
