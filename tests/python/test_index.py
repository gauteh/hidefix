from hidefix import Index

def test_index_coads(coads):
    i = Index(coads)
    print(str(i))

def test_dataset_coads(coads):
    i = Index(coads)
    ds = i.dataset('SST')
    print(ds)

