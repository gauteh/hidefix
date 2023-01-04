from hidefix import Index

def test_dataset_coads(coads):
    i = Index(coads)
    ds = i.dataset('SST') # this takes a couple of hundred nanoseconds.

    print(len(ds))

    arr = ds[0:10]
    print(arr)
