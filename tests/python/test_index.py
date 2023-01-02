from hidefix import Index

def test_index_coads(coads):
    i = Index(coads)
    print(i)

def test_dataset_coads(coads):
    i = Index(coads)
    print(i.dataset('SST'))

