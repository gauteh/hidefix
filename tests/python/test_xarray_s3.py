"""
Tests for the xarray backend reading from S3, against a local Minio instance.

Start Minio with:

    docker compose -f tests/docker-compose.minio.yml up -d

and run the tests with:

    HIDEFIX_S3_ENDPOINT=http://localhost:9000 pytest tests/python/test_xarray_s3.py

The tests are skipped when `HIDEFIX_S3_ENDPOINT` is not set (mirroring
`tests/read_s3.rs`); uploading the fixtures additionally needs `boto3`.
"""

import os

import numpy as np
import pytest
import xarray as xr

import hidefix

ENDPOINT = os.environ.get('HIDEFIX_S3_ENDPOINT')
ACCESS_KEY = os.environ.get('HIDEFIX_S3_ACCESS_KEY', 'minioadmin')
SECRET_KEY = os.environ.get('HIDEFIX_S3_SECRET_KEY', 'minioadmin')

BUCKET = 'hidefix-test'

# Tests that actually talk to the object store need a live endpoint; the
# constructor-only test below does not, so this is a per-test marker rather than
# a module-level `pytestmark`.
requires_endpoint = pytest.mark.skipif(
    ENDPOINT is None,
    reason='HIDEFIX_S3_ENDPOINT not set (start Minio with '
    'docker compose -f tests/docker-compose.minio.yml up -d)')


@pytest.fixture(scope='module')
def coads_path():
    # module-scoped twin of the function-scoped `coads` fixture in conftest.
    from pathlib import Path
    return Path(__file__).parent.parent / 'data' / 'coads_climatology.nc4'


@pytest.fixture(scope='module')
def coads_key(coads_path):
    # index the file and upload the object under the same (repo relative)
    # path, so the identity check has something to match.
    return os.path.relpath(coads_path)


@pytest.fixture(scope='module')
def bucket(coads_path, coads_key):
    """The test bucket with the coads fixture uploaded, under its repo
    relative path as key (like tests/read_s3.rs) and under a renamed key."""
    boto3 = pytest.importorskip('boto3')

    s3 = boto3.client('s3',
                      endpoint_url=ENDPOINT,
                      aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    try:
        s3.head_bucket(Bucket=BUCKET)
    except s3.exceptions.ClientError:
        s3.create_bucket(Bucket=BUCKET)

    for key in (coads_key, 'renamed/coads.nc4'):
        s3.upload_file(str(coads_path), BUCKET, key)

    return BUCKET


@pytest.fixture
def s3_kwargs():
    return dict(endpoint=ENDPOINT, access_key=ACCESS_KEY,
                secret_key=SECRET_KEY)


@pytest.fixture
def index(coads_key):
    return hidefix.Index(coads_key)


@requires_endpoint
def test_s3_equals_local(bucket, coads_path, coads_key, index, s3_kwargs):
    s3 = xr.open_dataset(f's3://{bucket}/{coads_key}', engine='hidefix',
                         index=index, decode_times=False, **s3_kwargs)
    local = xr.open_dataset(coads_path, engine='hidefix', decode_times=False)

    assert set(s3.data_vars) == set(local.data_vars)
    assert set(s3.coords) == set(local.coords)
    assert dict(s3.sizes) == dict(local.sizes)

    assert dict(s3.attrs) == dict(local.attrs)

    for name in local.variables:
        assert s3[name].dims == local[name].dims
        assert s3[name].dtype == local[name].dtype
        assert set(s3[name].attrs) == set(local[name].attrs)
        for k, v in local[name].attrs.items():
            np.testing.assert_array_equal(s3[name].attrs[k], v)
        np.testing.assert_array_equal(s3[name].values, local[name].values)


@requires_endpoint
def test_s3_slice(bucket, coads_path, coads_key, index, s3_kwargs):
    s3 = xr.open_dataset(f's3://{bucket}/{coads_key}', engine='hidefix',
                         index=index, decode_times=False, **s3_kwargs)
    local = xr.open_dataset(coads_path, engine='hidefix', decode_times=False)

    np.testing.assert_array_equal(
        s3['SST'][3:7, 10:80, 0:90].values,
        local['SST'][3:7, 10:80, 0:90].values)


@requires_endpoint
def test_s3_serialized_index(bucket, coads_path, coads_key, index, s3_kwargs,
                             tmp_path):
    p = tmp_path / 'coads.idx'
    index.save(p)

    s3 = xr.open_dataset(f's3://{bucket}/{coads_key}', engine='hidefix',
                         index=p, decode_times=False, **s3_kwargs)
    local = xr.open_dataset(coads_path, engine='hidefix', decode_times=False)
    np.testing.assert_array_equal(s3['SST'].values, local['SST'].values)


@requires_endpoint
def test_s3_open_is_lazy(bucket, coads_key, index, s3_kwargs, monkeypatch):
    # opening must not fetch any data variable: xarray only loads the (small)
    # dimension coordinate variables to build its indexes, everything else
    # all comes from the local index. Indexing a variable reads just that
    # variable, with just the requested key.
    from hidefix.xarray import HidefixArray

    reads = []
    orig = HidefixArray._getitem

    def traced(self, key):
        reads.append((self.variable_name, key))
        return orig(self, key)

    monkeypatch.setattr(HidefixArray, '_getitem', traced)

    ds = xr.open_dataset(f's3://{bucket}/{coads_key}', engine='hidefix',
                         index=index, decode_times=False, **s3_kwargs)
    assert set(v for v, _ in reads) == {'COADSX', 'COADSY', 'TIME'}

    n = len(reads)
    ds['SST'][0, 0:2, 0:2].values
    assert reads[n:] == [('SST', (0, slice(0, 2, 1), slice(0, 2, 1)))]


@requires_endpoint
def test_s3_requires_index(bucket, coads_key, s3_kwargs):
    with pytest.raises(ValueError, match='requires index='):
        xr.open_dataset(f's3://{bucket}/{coads_key}', engine='hidefix',
                        decode_times=False, **s3_kwargs)


@requires_endpoint
def test_s3_index_identity(bucket, index, s3_kwargs):
    # the index's source_path does not match the renamed key: refused under
    # 'verify' (the default), allowed under 'ignore'.
    with pytest.raises(ValueError, match='matches neither'):
        xr.open_dataset(f's3://{bucket}/renamed/coads.nc4', engine='hidefix',
                        index=index, decode_times=False, **s3_kwargs)

    ds = xr.open_dataset(f's3://{bucket}/renamed/coads.nc4', engine='hidefix',
                         index=index, index_fingerprint='ignore',
                         decode_times=False, **s3_kwargs)
    assert np.isfinite(ds['SST'].values).any()


@requires_endpoint
def test_s3_kwargs_rejected_for_local_paths(coads_path):
    with pytest.raises(ValueError, match='only valid'):
        xr.open_dataset(coads_path, engine='hidefix', decode_times=False,
                        endpoint='http://localhost:9000')


@requires_endpoint
def test_s3_anonymous_rejected(bucket, coads_key, index, s3_kwargs):
    # minio requires signed requests by default: anonymous access must fail
    # once something is read (the coordinate variables, at open), proving the
    # flag switches off request signing.
    with pytest.raises(Exception, match='(?i)range request|status code'):
        xr.open_dataset(f's3://{bucket}/{coads_key}', engine='hidefix',
                        index=index, decode_times=False,
                        endpoint=ENDPOINT, anonymous=True)


@pytest.mark.skipif(not hasattr(hidefix, 'S3Source'),
                    reason='hidefix built without the s3 feature')
@pytest.mark.parametrize('cred', ['access_key', 'secret_key', 'session_token'])
def test_s3source_anonymous_excludes_explicit_credentials(cred):
    # anonymous and explicit credentials are mutually exclusive: combining
    # them used to silently win for anonymous, dropping valid keys and issuing
    # an unsigned request. Construction must reject the combination (no live
    # endpoint needed). The valid explicit-credentials path is exercised by the
    # endpoint tests above.
    with pytest.raises(ValueError, match='(?i)mutually exclusive|anonymous'):
        hidefix.S3Source('bucket', 'key', region='us-east-1', anonymous=True,
                         **{cred: 'x'})

    # sanity: anonymous alone, and explicit credentials alone, both construct.
    hidefix.S3Source('bucket', 'key', region='us-east-1', anonymous=True)
    hidefix.S3Source('bucket', 'key', region='us-east-1',
                     access_key='a', secret_key='b')
