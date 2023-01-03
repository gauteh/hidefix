from netCDF4 import Dataset

def test_nc_open_coads(coads, benchmark):
    i = benchmark(Dataset, coads)
    print(str(i))

def test_nc_open_large(large_file, benchmark):
    f, v = large_file
    i = benchmark(Dataset, f)
    print(str(i))
