import pytest

@pytest.fixture
def data():
    from pathlib import Path
    return Path(__file__).parent / 'data'

@pytest.fixture
def coads():
    from pathlib import Path
    return Path(__file__).parent / 'data' / 'coads_climatology.nc4'

