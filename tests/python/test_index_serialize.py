import gc
import os
import shutil
import struct

import numpy as np
import pytest
import xarray as xr

from hidefix import Index
from hidefix.xarray import HidefixDataStore

from test_xarray_backend import assert_engines_equal


def test_save_load_roundtrip(coads, tmp_path):
    idx = Index(coads)
    st = os.stat(coads)
    assert idx.source_path == str(coads)
    assert idx.source_size == st.st_size
    assert idx.source_mtime == int(st.st_mtime)

    p = tmp_path / 'coads.idx'
    idx.save(p)
    li = Index.load_index(p)

    assert li.source_path == idx.source_path
    assert li.source_size == idx.source_size
    assert li.source_mtime == idx.source_mtime
    assert set(li.datasets(None)) == set(idx.datasets(None))
    assert li.dataset_dims('SST', None) == idx.dataset_dims('SST', None)
    assert li.dataset_attributes('SST',
                                 None) == idx.dataset_attributes('SST', None)
    np.testing.assert_array_equal(li['SST'][()], idx['SST'][()])


def test_bytes_roundtrip(coads):
    idx = Index(coads)
    b = idx.to_bytes()
    assert isinstance(b, bytes)

    li = Index.load_index(b)
    assert li.source_size == idx.source_size
    assert li.source_mtime == idx.source_mtime
    assert set(li.datasets(None)) == set(idx.datasets(None))


def test_load_index_rejects_other_types():
    with pytest.raises(TypeError):
        Index.load_index(42)


def test_open_dataset_with_index(coads, tmp_path):
    idx = Index(coads)
    p = tmp_path / 'coads.idx'
    idx.save(p)

    fresh = xr.open_dataset(coads, engine='hidefix', decode_times=False)

    # an Index object, a path to a serialized index, and its bytes.
    for index in (idx, p, idx.to_bytes()):
        ds = xr.open_dataset(coads,
                             engine='hidefix',
                             decode_times=False,
                             index=index)
        assert ds.identical(fresh)

    # opening with a serialized index must also match the netcdf4 engine.
    assert_engines_equal(coads,
                         decode_times=False,
                         hidefix_kwargs={'index': p})


def test_stale_index_rejected(coads, tmp_path):
    f = tmp_path / 'coads.nc4'
    shutil.copy(coads, f)
    idx = Index(f)
    b = idx.to_bytes()

    # touch the file: same content, newer mtime -> stale index.
    st = os.stat(f)
    os.utime(f, (st.st_atime, st.st_mtime + 10))

    with pytest.raises(ValueError, match='stale'):
        xr.open_dataset(f, engine='hidefix', decode_times=False, index=b)

    # ignore-mode opens anyway (the content is unchanged).
    ds = xr.open_dataset(f,
                         engine='hidefix',
                         decode_times=False,
                         index=b,
                         index_fingerprint='ignore')
    ncd = xr.open_dataset(f, engine='netcdf4', decode_times=False)
    np.testing.assert_array_equal(ds['SST'], ncd['SST'])


def test_index_for_other_file_rejected(coads, tmp_path):
    other = tmp_path / 'other.nc4'
    shutil.copy(coads, other)
    idx = Index(coads)

    with pytest.raises(ValueError, match='built from'):
        xr.open_dataset(other, engine='hidefix', decode_times=False, index=idx)

    # 'ignore' only skips the staleness check: reads go through the indexed
    # path, so a wrong-file index would silently return the other file's data.
    with pytest.raises(ValueError, match='built from'):
        xr.open_dataset(other,
                        engine='hidefix',
                        decode_times=False,
                        index=idx,
                        index_fingerprint='ignore')


def test_size_changed_index_rejected(coads, tmp_path):
    f = tmp_path / 'coads.nc4'
    shutil.copy(coads, f)
    b = Index(f).to_bytes()

    # append a byte: the size no longer matches the fingerprint.
    with open(f, 'ab') as fd:
        fd.write(b'\0')

    with pytest.raises(ValueError, match='size/mtime'):
        xr.open_dataset(f, engine='hidefix', decode_times=False, index=b)


def test_load_index_owns_bytes(coads):
    idx = Index(coads)
    expected = idx['SST'][()]

    b = idx.to_bytes()
    li = Index.load_index(b)
    del b, idx
    gc.collect()

    np.testing.assert_array_equal(li['SST'][()], expected)

    ds = xr.open_dataset(coads, engine='hidefix', decode_times=False, index=li)
    assert ds['SST'].shape == expected.shape


def test_invalid_fingerprint_mode_rejected(coads):
    idx = Index(coads)
    with pytest.raises(ValueError, match='index_fingerprint'):
        xr.open_dataset(coads,
                        engine='hidefix',
                        decode_times=False,
                        index=idx,
                        index_fingerprint='bogus')


def test_unknown_format_version_rejected(coads):
    b = Index(coads).to_bytes()
    assert b[:4] == b'HFXI'
    bad = b[:4] + struct.pack('<I', 99) + b[8:]

    with pytest.raises(RuntimeError, match='format version 99'):
        Index.load_index(bad)


def test_garbage_bytes_rejected():
    with pytest.raises(RuntimeError, match='magic'):
        Index.load_index(b'not an index at all')


class _MetalessIndex:
    """Proxy behaving like an index serialized before hidefix captured dataset
    metadata: dataset lookups succeed, metadata lookups raise KeyError. The
    real deserialization of such an index is covered on the Rust side
    (idx::index::tests::deserialize_index_without_metadata_fields)."""

    def __init__(self, idx):
        self._idx = idx

    def __getattr__(self, name):
        return getattr(self._idx, name)

    def dataset_attributes(self, *args):
        raise KeyError('dataset not found: SST')

    def dataset_dims(self, *args):
        raise KeyError('dataset not found: SST')


def test_old_index_without_metadata_clear_error(coads):
    store = HidefixDataStore(str(coads))
    store.idx = _MetalessIndex(store.idx)

    with pytest.raises(ValueError, match='older hidefix'):
        store.open_store_variable('SST')
