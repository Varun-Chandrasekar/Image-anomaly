"""Independent verification: Fiji and Python must agree on simple metrics."""
import numpy as np
import pytest

from src import fiji_check
from src.io_utils import load_still
from src.qc_metrics import image_metrics

pytestmark = [pytest.mark.fiji, pytest.mark.skipif(not fiji_check.available(), reason="PyImageJ not installed")]


@pytest.mark.parametrize("name", ["camera", "coins", "moon"])
def test_fiji_matches_python(name):
    img = load_still(name)
    py, fj = image_metrics(img), fiji_check.measure(img)
    assert abs(py["mean"] - fj["mean"]) < 2 / 255
    assert abs(py["saturated_frac"] - fj["saturated_frac"]) < 0.01
    assert abs(py["dark_frac"] - fj["dark_frac"]) < 0.01
