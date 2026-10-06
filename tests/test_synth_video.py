import numpy as np

from src.io_utils import load_still
from src.synth_video import make_blob_video, make_lc_video


def test_blob_video_labels_match_injected_faults():
    v, labels, meta = make_blob_video(load_still("camera"), seed=0)
    assert v.shape == (200, 128, 128)
    assert (labels.fault.iloc[80:91] == "gain").all()
    assert labels.fault.iloc[120] == "jitter"
    assert labels.fault.iloc[150] == "duplicate"
    np.testing.assert_array_equal(v[150], v[149])
    assert meta["stuck_mask"].sum() == 20
    assert labels.has_event.any() and not labels.has_event.all()


def test_lc_video_uses_real_background():
    rng = np.random.default_rng(1)
    real = np.clip(0.4 + 0.05 * rng.standard_normal((20, 48, 64)), 0, 1)
    v, labels, meta = make_lc_video(real, n_frames=30, seed=0, faults=False)
    assert v.shape == (30, 48, 64)
    assert abs(np.median(v) - 0.4) < 0.05
    assert meta["noise_sigma"] > 0
