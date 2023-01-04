import os
from pathlib import Path
import pytest

@pytest.fixture
def data():
    return Path(__file__).parent / 'data'

@pytest.fixture
def coads():
    return Path(__file__).parent / 'data' / 'coads_climatology.nc4'

@pytest.fixture
def feb():
    return Path(__file__).parent / 'data' / 'feb.nc4'

@pytest.fixture
def large_file():
    f = os.getenv('HIDEFIX_LARGE_FILE')
    v = os.getenv('HIDEFIX_LARGE_VAR')

    if f is None or v is None:
        pytest.skip("No HIDEFIX_LARGE_FILE or HIDEFIX_LARGE_VAR specified")

    return Path(f), v

def pytest_addoption(parser):
    parser.addoption(
            "--plot", action="store_true", default=False, help="show plots"
            )

@pytest.fixture
def plot(pytestconfig):
    return pytestconfig.getoption('plot')
