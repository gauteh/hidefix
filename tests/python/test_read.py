from operator import getitem
from hidefix import Index

def test_read_large_hidefix(large_file, benchmark):
    f, v = large_file
    i = Index(f)
    var = i[v]

    values = benchmark(getitem, var, tuple())
    print(values.shape, type(values))


