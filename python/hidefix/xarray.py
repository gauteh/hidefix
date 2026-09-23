"""
An Xarray backend based on hidefix.

Datasets are opened from a local file, or -- when built with the `s3` feature
-- directly from S3 (or an S3-compatible object store) with an `s3://` URI.
Opening from S3 differs from a local open:

* An `index` is required: indexing itself needs a local file, so the index
  must be built from a local copy of the object (`hidefix.Index(path)`) and
  passed in, either live or serialized (`Index.save` / `Index.to_bytes`).
* Identity is checked by string: the index's embedded `source_path` must match
  the object key (or the full URI) exactly (`os.path.samefile` is meaningless
  against an object store). The check is skipped with
  `index_fingerprint='ignore'`, e.g. when the object was renamed on upload.
  Size/mtime staleness cannot be verified remotely and is not checked. An index
  built without a source path (`Index.source_path` is None) has nothing to
  match, so its identity is silently not verified even under 'verify' (as with
  the local open).
* Reads are lazy: all metadata comes from the index, and opening fetches only
  the (small) dimension coordinate variables xarray loads to build its
  indexes. Data variable chunk ranges are fetched (with coalesced, concurrent
  range requests) when a variable is actually indexed.
"""

import os
import logging

import hidefix
import numpy as np

from xarray.backends.common import (
    BackendArray,
    BackendEntrypoint,
    WritableCFDataStore,
    _normalize_path,
)

from xarray.core import indexing
from xarray.backends.store import StoreBackendEntrypoint
from xarray.coding.variables import pop_to
from xarray.core.variable import Variable

from xarray.core.utils import FrozenDict

logger = logging.getLogger(__name__)


class HidefixBackendEntrypoint(BackendEntrypoint):

    available = True
    description = "Open netCDF4 files with the multi-threaded Hidefix backend in Xarray"
    url = "https://github.com/gauteh/hidefix"
    open_dataset_parameters = [
        "filename_or_obj", "drop_variables", "index", "index_fingerprint",
        "region", "endpoint", "anonymous", "access_key", "secret_key",
        "session_token", "path_style"
    ]

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        index=None,
        index_fingerprint='verify',
        region=None,
        endpoint=None,
        anonymous=False,
        access_key=None,
        secret_key=None,
        session_token=None,
        path_style=None,
    ):
        """Open a netCDF4/HDF5 file with the hidefix engine.

        `index` skips re-indexing the file: either a `hidefix.Index`, or a
        path to / the bytes of one serialized with `Index.save` /
        `Index.to_bytes`. The index's path identity is always verified against
        the file (reads go through the indexed path, so a wrong-file index
        would silently return the other file's data); pass
        `index_fingerprint='ignore'` to skip only the size/mtime staleness
        check ('verify' is the default).

        With an `s3://bucket/key` URI the chunks are fetched from the object
        store instead (see the module docstring for how this differs from a
        local open). `index` is then required, and identity is checked by
        exact string match of the index's `source_path` against the object
        key or the full URI ('verify', the default), or not at all
        ('ignore'). `region`, `endpoint`, `anonymous`,
        `access_key`/`secret_key`/`session_token` and `path_style` configure
        the connection, see `hidefix.S3Source`.
        """
        filename_or_obj = _normalize_path(filename_or_obj)

        if index_fingerprint not in ('verify', 'ignore'):
            raise ValueError(
                "index_fingerprint must be 'verify' or 'ignore', got: "
                f"{index_fingerprint!r}")

        if index is not None and not isinstance(index, hidefix.Index):
            index = hidefix.Index.load_index(index)

        s3_kwargs = {
            'region': region,
            'endpoint': endpoint,
            'anonymous': anonymous,
            'access_key': access_key,
            'secret_key': secret_key,
            'session_token': session_token,
            'path_style': path_style,
        }

        if _is_s3_uri(filename_or_obj):
            s3 = _s3_source(filename_or_obj, index, **s3_kwargs)
            if index_fingerprint == 'verify':
                _verify_s3_index_identity(index, filename_or_obj, s3.key)
        else:
            s3 = None
            if any(v for v in s3_kwargs.values()):
                raise ValueError(
                    "S3 arguments (region, endpoint, ...) are only valid "
                    f"with an s3:// uri, not {filename_or_obj!r}")
            if index is not None:
                _verify_index_fingerprint(
                    index,
                    filename_or_obj,
                    check_staleness=index_fingerprint == 'verify')

        store = HidefixDataStore.open(filename_or_obj, group, index, s3)

        store_entrypoint = StoreBackendEntrypoint()
        return store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

    def guess_can_open(self, filename_or_obj):
        # both local paths and s3:// uris, by extension.
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".nc", ".nc4", ".cdf"}


def _is_s3_uri(filename):
    return isinstance(filename, str) and filename.startswith('s3://')


def _s3_source(uri, index, **kwargs):
    """A `hidefix.S3Source` for `s3://bucket/key`, requiring `index` (indexing
    needs a local file, so it cannot be built from the object itself)."""
    if not hasattr(hidefix, 'S3Source'):
        raise ValueError(
            "this hidefix build does not support S3 (built without the "
            "'s3' feature)")

    if index is None:
        raise ValueError(
            f"opening {uri!r} requires index=: indexing needs a local file, "
            "build the index from a local copy of the object "
            "(hidefix.Index(path)) and pass it, live or serialized "
            "(Index.save / Index.to_bytes)")

    bucket, _, key = uri[len('s3://'):].partition('/')
    if not bucket or not key:
        raise ValueError(f"invalid s3 uri (expected s3://bucket/key): {uri!r}")

    return hidefix.S3Source(bucket, key, **kwargs)


def _verify_s3_index_identity(index, uri, key):
    """Raise ValueError when the index's embedded source path matches neither
    the object key nor the full uri, by exact string comparison (`samefile`
    is meaningless against an object store, and size/mtime staleness cannot
    be verified remotely). Skipped under `index_fingerprint='ignore'`."""
    source = index.source_path
    if source is not None and source not in (uri, key):
        raise ValueError(
            f"index was built from {source!r} which matches neither the "
            f"object key {key!r} nor {uri!r}; if this is the same file "
            "renamed, pass index_fingerprint='ignore', otherwise re-index "
            "a local copy of the object")


def _verify_index_fingerprint(index, filename, check_staleness=True):
    """Raise ValueError when `index` does not match `filename`: identity by
    path (always checked -- reads go through the indexed path), staleness by
    size + mtime only when `check_staleness` (a size or mtime of 0 means
    unknown and is not checked)."""
    source = index.source_path
    if source is not None and os.path.exists(source) and os.path.exists(
            filename) and not os.path.samefile(source, filename):
        raise ValueError(
            f"index was built from {source!r}, not {filename!r} (reads go "
            "through the indexed path); re-index the file")

    if not check_staleness:
        return

    if index.source_size == 0 or index.source_mtime == 0:
        return

    st = os.stat(filename)
    if st.st_size != index.source_size or int(
            st.st_mtime) != index.source_mtime:
        raise ValueError(
            f"index is stale for {filename!r}: indexed size/mtime "
            f"({index.source_size}, {index.source_mtime}) does not match the "
            f"file ({st.st_size}, {int(st.st_mtime)}); re-index the file or "
            "pass index_fingerprint='ignore'")


class HidefixDataStore(WritableCFDataStore):
    idx: hidefix.Index
    path: str
    group: str

    def __init__(self, path, group=None, index=None, s3=None):
        self.path = path
        self.group = group
        self.s3 = s3

        self.idx = hidefix.Index(path) if index is None else index

    @classmethod
    def open(
        cls,
        filename,
        group,
        index=None,
        s3=None,
    ):
        if isinstance(filename, os.PathLike):
            filename = os.fspath(filename)

        if not isinstance(filename, str):
            raise ValueError(
                "the hidefix backend can only read file-like objects")

        return cls(filename, group, index, s3)

    def dataset(self, name, group):
        """The dataset handle reads are issued through: S3-backed when the
        store was opened with an `s3://` uri, otherwise the local indexed
        path."""
        ds = self.idx.dataset(name, group)
        if ds is not None and self.s3 is not None:
            ds = ds.with_s3(self.s3)
        return ds

    def get_attrs(self):
        return FrozenDict(self.idx.attributes(self.group))

    def get_dimensions(self):
        # dimension sizes from the per-variable dims/shape zip; a dimension
        # scale names itself, so coordinate variables are covered too.
        dims = {}
        for name in self.idx.datasets(self.group):
            ds = self.idx.dataset(name, self.group)
            for dim, size in zip(self.idx.dataset_dims(name, self.group),
                                 ds.shape()):
                dims[dim] = int(size)
        return FrozenDict(dims)

    def get_encoding(self):
        # unlimited dimensions are not captured by the index.
        return {"unlimited_dims": set()}

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k)) for k in self.idx.datasets(self.group))

    def open_store_variable(self, k):
        ds = self.idx.dataset(k, self.group)
        try:
            attributes = self.idx.dataset_attributes(k, self.group)
            dimensions = tuple(self.idx.dataset_dims(k, self.group))
        except KeyError:
            # only reachable with a loaded index serialized before hidefix
            # captured attributes and dimension names.
            raise ValueError(
                f"index has no metadata for variable {k!r}: it was likely "
                "serialized by an older hidefix; re-index the file with a "
                "current hidefix (hidefix.Index(path))") from None

        data = indexing.LazilyIndexedArray(
            HidefixArray(self, k, self.group, ds, attributes))

        # fill values are applied (as NaN) by HidefixArray.
        attributes.pop('_FillValue', None)
        attributes.pop('missing_value', None)

        encoding = {}
        pop_to(attributes, encoding, "least_significant_digit")
        # save source so __repr__ can detect if it's local or not
        encoding["source"] = self.path
        encoding["original_shape"] = data.shape
        encoding["dtype"] = data.dtype

        return Variable(dimensions, data, attributes, encoding)


class HidefixArray(BackendArray):

    def __init__(self, store, name, group, ds, attributes):
        self.store = store
        self.variable_name = name
        self.group = group

        self.shape = tuple(int(s) for s in ds.shape())
        self.dtype = np.dtype(ds.dtype())
        self.fill_value = attributes.get('_FillValue', None)
        missing = attributes.get('missing_value', None)
        if missing is not None:
            if self.fill_value is None:
                self.fill_value = missing
            else:
                assert missing == self.fill_value, "mismatch between missing_value and _FillValue"

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem)

    def _getitem(self, key):
        #TODO: perf: cache this? maybe this is making single value access slow.
        array = self.store.dataset(self.variable_name, self.group)
        data = array[key]
        if self.fill_value is not None:
            array.apply_fill_value(self.fill_value, np.nan, data)
        return data
