import numpy as np

from src import artefacts
from src.io_utils import load_still
from src.qc_metrics import image_metrics, stuck_pixels, video_metrics
from src.synth_video import make_blob_video

IMG = load_still("camera")


def _metric_vs_level(name, metric):
    return [image_metrics(artefacts.apply(name, IMG, l, np.random.default_rng(0)))[metric] for l in range(1, 5)]


def test_each_metric_tracks_its_artefact():
    assert np.all(np.diff(_metric_vs_level("gaussian_noise", "noise_sigma")) > 0)
    assert np.all(np.diff(_metric_vs_level("gaussian_blur", "sharpness")) < 0)
    assert np.all(np.diff(_metric_vs_level("column_banding", "banding_col")) > 0)
    over = image_metrics(artefacts.exposure(IMG, 2.5))["saturated_frac"]
    assert over > image_metrics(IMG)["saturated_frac"] + 0.2


def test_video_metrics_find_injected_faults():
    v, labels, meta = make_blob_video(load_still("camera"), seed=0)
    vm = video_metrics(v)
    assert vm.is_duplicate.iloc[150]
    assert vm["shift"].iloc[120] > 1.5          # jitter of 2 px
    jump = np.diff(vm.brightness.values)
    assert np.argmax(jump) == 79                # gain starts at frame 80
    np.testing.assert_array_equal(stuck_pixels(v) & meta["stuck_mask"], meta["stuck_mask"])
