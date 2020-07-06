import os
from h5py import File
import numpy as np

# http://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline

f = File('gzip_shuffle_2d.h5', 'w')

data = f.create_dataset('data', (10000, 100), compression = 'gzip', shuffle = True, chunks = (100, 10))
data[:] = np.random.rand(10000, 100)

