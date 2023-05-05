"""
An Xarray backend based on hidefix and netCDF4.
"""

import os
from pathlib import Path
import logging
import operator

import hidefix
import netCDF4 as nc
import xarray as xr
import numpy as np

from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    BackendArray,
    BackendEntrypoint,
    WritableCFDataStore,
    _normalize_path,
    find_root_and_group,
    robust_getitem,
)

from xarray.core import indexing
from xarray.backends.store import StoreBackendEntrypoint
from xarray.coding.variables import pop_to
from xarray.core.variable import Variable

from xarray.core.utils import (
    FrozenDict,
    close_on_error,
    is_remote_uri,
    try_read_magic_number_from_path,
)

logger = logging.getLogger(__name__)


class HidefixBackendEntrypoint(BackendEntrypoint):

    available = True
    description = "Open netCDF4 files with the multi-threaded Hidefix backend in Xarray"
    url = "https://github.com/gauteh/hidefix"
    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

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
    ):
        # TODO: allow take an existing index, maybe from a serialized object.

        filename_or_obj = _normalize_path(filename_or_obj)

        store = HidefixDataStore.open(filename_or_obj)

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
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".nc", ".nc4", ".cdf"}


class HidefixDataStore(WritableCFDataStore):
    idx: hidefix.Index
    path: Path
    ds: nc.Dataset

    def __init__(self, path):
        self.path = path

        # These can be done concurrently
        self.idx = hidefix.Index(path)
        self.ds = nc.Dataset(path, mode='r')

    @classmethod
    def open(
        cls,
        filename,
    ):
        if isinstance(filename, os.PathLike):
            filename = os.fspath(filename)

        if not isinstance(filename, str):
            raise ValueError(
                "the hidefix backend can only read file-like objects")

        return cls(filename)

    def get_attrs(self):
        return FrozenDict((k, self.ds.getncattr(k)) for k in self.ds.ncattrs())

    def get_dimensions(self):
        return FrozenDict((k, len(v)) for k, v in self.ds.dimensions.items())

    def get_encoding(self):
        return {
            "unlimited_dims":
            {k
             for k, v in self.ds.dimensions.items() if v.isunlimited()}
        }

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k)) for k in self.idx.datasets())

    def open_store_variable(self, k):
        var = self.ds.variables[k]
        attributes = {k: var.getncattr(k) for k in var.ncattrs()}

        data = indexing.LazilyIndexedArray(
            HidefixArray(self, k, var, attributes))

        attributes.pop('_FillValue', None)
        attributes.pop('missing_value', None)

        dimensions = var.dimensions
        xr.backends.netCDF4_._ensure_fill_value_valid(data, attributes)
        encoding = {}
        filters = var.filters()
        if filters is not None:
            encoding.update(filters)
        chunking = var.chunking()
        if chunking is not None:
            if chunking == "contiguous":
                encoding["contiguous"] = True
                encoding["chunksizes"] = None
            else:
                encoding["contiguous"] = False
                encoding["chunksizes"] = tuple(chunking)
        # TODO: figure out how to round-trip "endian-ness" without raising
        # warnings from netCDF4
        # encoding['endian'] = var.endian()
        pop_to(attributes, encoding, "least_significant_digit")
        # save source so __repr__ can detect if it's local or not
        encoding["source"] = self.path
        encoding["original_shape"] = var.shape
        encoding["dtype"] = var.dtype

        return Variable(dimensions, data, attributes, encoding)


class HidefixArray(BackendArray):

    def __init__(self, store, name, var, attributes):
        self.store = store
        self.variable_name = name

        self.shape = var.shape
        self.dtype = var.dtype
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
        array = self.store.idx[self.variable_name]
        data = array[key]
        if self.fill_value is not None:
            array.apply_fill_value(self.fill_value, np.nan, data)
        return data

