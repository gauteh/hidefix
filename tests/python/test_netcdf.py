from operator import getitem
from netCDF4 import Dataset

def test_nc_open_coads(coads, benchmark):
    i = benchmark(Dataset, coads)
    print(str(i))

def test_nc_open_large(large_file, benchmark):
    f, v = large_file
    i = benchmark(Dataset, f)
    print(str(i))

def test_read_large_netcdf4(large_file, benchmark):
    f, v = large_file
    d = Dataset(f)
    var = d[v]

    values = benchmark(getitem, var, ...)
    print(values.shape, type(values))
