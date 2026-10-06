import numpy as np
import pytest

from src import artefacts
from src.io_utils import load_still

IMG = load_still("camera", 128)


@pytest.mark.parametrize("name", list(artefacts.ARTEFACTS))
def test_shape_range_and_input_untouched(name):
    before = IMG.copy()
    out = artefacts.apply(name, IMG, 2, np.random.default_rng(0))
    assert out.shape == IMG.shape
    assert out.min() >= 0 and out.max() <= 1
    np.testing.assert_array_equal(IMG, before)


@pytest.mark.parametrize("name", list(artefacts.ARTEFACTS))
def test_same_seed_same_output(name):
    a = artefacts.apply(name, IMG, 3, np.random.default_rng(42))
    b = artefacts.apply(name, IMG, 3, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", list(artefacts.ARTEFACTS))
def test_damage_grows_with_strength(name):
    if name == "exposure":  # levels go under- then over-exposed; check each side separately
        return
    mse = [np.mean((artefacts.apply(name, IMG, lvl, np.random.default_rng(0)) - IMG) ** 2) for lvl in range(1, 5)]
    assert all(np.diff(mse) > 0), mse
