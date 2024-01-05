import pytest
import numpy as np



@pytest.fixture
def random_generator():
    return np.random.default_rng(seed=42)
