import numpy as np

from src import artefacts
from src.features import blockiness, wavelet_denoise, wavelet_energies
from src.io_utils import load_still

IMG = load_still("camera")
RNG = lambda: np.random.default_rng(0)


def test_energies_are_shares():
    e = wavelet_energies(IMG)
    assert abs(sum(e.values()) - 1) < 1e-9
    assert len(e) == 1 + 3 * 3


def test_noise_raises_and_blur_lowers_finest_detail():
    fine = lambda im: sum(wavelet_energies(im)[f"wav_{o}3"] for o in "HVD")
    assert fine(artefacts.gaussian_noise(IMG, 0.1, RNG())) > fine(IMG)
    assert fine(artefacts.gaussian_blur(IMG, 3.0)) < fine(IMG)


def test_column_banding_adds_vertical_detail():
    band = artefacts.column_banding(IMG, 0.1, RNG())
    assert wavelet_energies(band)["wav_V3"] > 2 * wavelet_energies(IMG)["wav_V3"]


def test_jpeg_blockiness():
    assert blockiness(artefacts.jpeg(IMG, 5)) > blockiness(IMG) + 0.2


def test_denoise_reduces_error():
    noisy = artefacts.gaussian_noise(IMG, 0.1, RNG())
    assert np.mean((wavelet_denoise(noisy) - IMG) ** 2) < np.mean((noisy - IMG) ** 2)
