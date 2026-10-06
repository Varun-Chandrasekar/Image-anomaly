"""Artefact injection. Every function has the signature f(img, strength, rng) -> img.

Inputs and outputs are float64 grayscale images in [0, 1]; the input is never modified.
Because we inject the damage ourselves, the label (artefact, strength) is exact.
"""
from __future__ import annotations

import cv2
import numpy as np


def gaussian_noise(img, sigma, rng):
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 1)


def salt_pepper(img, frac, rng):
    out = img.copy()
    m = rng.random(img.shape) < frac
    out[m] = rng.choice([0.0, 1.0], m.sum())
    return out


def gaussian_blur(img, sigma, rng=None):
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma).astype(np.float64)


def motion_blur(img, length, rng=None):
    """Horizontal motion blur with a line kernel of the given length (pixels)."""
    length = int(length)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    return cv2.filter2D(img.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT).astype(np.float64)


def dead_pixels(img, frac, rng):
    """Dead (0) or hot (1) pixels at random positions."""
    return salt_pepper(img, frac, rng)


def column_banding(img, sigma, rng):
    """Fixed-pattern noise: a random offset added to every column."""
    return np.clip(img + rng.normal(0, sigma, (1, img.shape[1])), 0, 1)


def exposure(img, gain, rng=None):
    """Over-exposure (gain > 1) or under-exposure (gain < 1), with clipping."""
    return np.clip(img * gain, 0, 1)


def jpeg(img, quality, rng=None):
    """JPEG compression at the given quality (lower = stronger artefact)."""
    u8 = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0


# name -> (function, strength levels from mild to severe)
ARTEFACTS = {
    "gaussian_noise": (gaussian_noise, [0.02, 0.05, 0.1, 0.2]),
    "salt_pepper": (salt_pepper, [0.005, 0.01, 0.03, 0.08]),
    "gaussian_blur": (gaussian_blur, [0.75, 1.5, 3.0, 5.0]),
    "motion_blur": (motion_blur, [3, 7, 13, 21]),
    "dead_pixels": (dead_pixels, [0.001, 0.003, 0.01, 0.03]),
    "column_banding": (column_banding, [0.01, 0.03, 0.06, 0.1]),
    "exposure": (exposure, [0.3, 0.6, 1.6, 2.5]),
    "jpeg": (jpeg, [50, 25, 10, 5]),
}


def apply(name: str, img: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply artefact `name` at strength level 1..4."""
    fn, levels = ARTEFACTS[name]
    return fn(img, levels[level - 1], rng)
