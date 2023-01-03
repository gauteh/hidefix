from hidefix import Index

def test_index_coads(coads, benchmark):
    i = benchmark(Index, coads)
    print(str(i))

def test_dataset_coads(coads):
    i = Index(coads)
    ds = i.dataset('SST') # this takes a couple of hundred nanoseconds.
    print(ds)

def test_index_large(large_file, benchmark):
    f, v = large_file
    i = benchmark(Index, f)
    print(str(i))

