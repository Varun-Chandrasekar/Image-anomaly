"""Build the labelled image set: clean images plus every artefact at every level."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import artefacts
from .io_utils import load_stills

TEST_SOURCES = ("text", "grass", "astronaut")  # held out; never seen in training


def build_dataset(seed: int = 0, size: int = 256, test_sources=TEST_SOURCES):
    """Return (images, table). images[i] is described by row i of table.

    Columns: source, artefact ("clean" for originals), level (0 = clean), strength, split.
    The split is by source image, so no source appears in both train and test.
    """
    rng = np.random.default_rng(seed)
    images, rows = [], []
    for src, img in load_stills(size=size).items():
        split = "test" if src in test_sources else "train"
        images.append(img)
        rows.append(dict(source=src, artefact="clean", level=0, strength=0.0, split=split))
        for name, (_, levels) in artefacts.ARTEFACTS.items():
            for lvl in range(1, len(levels) + 1):
                images.append(artefacts.apply(name, img, lvl, rng))
                rows.append(dict(source=src, artefact=name, level=lvl,
                                 strength=float(levels[lvl - 1]), split=split))
    return np.stack(images), pd.DataFrame(rows)


def clean_augment(seed: int = 0, n_per_source: int = 8, size: int = 256, test_sources=TEST_SOURCES,
                  split: str = "train"):
    """Extra clean images: random crops and flips of the larger originals.

    split="train" uses only training sources (what unsupervised detectors learn "normal" from);
    split="test" uses only held-out sources (more clean negatives for a fairer test AUC).
    """
    rng = np.random.default_rng(seed)
    out = []
    for src, img in load_stills(size=size * 2).items():
        if (src in test_sources) != (split == "test"):
            continue
        for _ in range(n_per_source):
            y, x = rng.integers(0, size + 1, 2)
            crop = img[y:y + size, x:x + size]
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            out.append(crop.copy())
    return np.stack(out)
