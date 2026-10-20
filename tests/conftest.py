import numpy as np
import pytest

from legobot.config import load_config
from legobot.localisation import arena


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def world():
    return arena()


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
