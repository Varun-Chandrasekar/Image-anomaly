"""Independent cross-check of selected metrics in Fiji (ImageJ), via PyImageJ.

Fiji is optional: everything else in the project runs without it. Install with
`pip install pyimagej` plus a Java runtime (OpenJDK 11+). The same macros live in
fiji/macros/ so they can also be run by hand in the Fiji GUI.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MACROS = Path(__file__).resolve().parent.parent / "fiji" / "macros"


def available() -> bool:
    try:
        import imagej  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def get_ij():
    import imagej
    return imagej.init("sc.fiji:fiji", mode="headless")


def measure(img: np.ndarray) -> dict:
    """Mean and saturated/dark fractions computed by Fiji on an 8-bit copy of `img`."""
    ij = get_ij()
    u8 = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
    imp = ij.py.to_imageplus(ij.py.to_java(u8))
    ij.WindowManager.setTempCurrentImage(imp)
    args = {"image": imp}
    result = ij.py.run_macro((MACROS / "measure.ijm").read_text(), args)
    return dict(mean=float(result.getOutput("mean")) / 255.0,
                saturated_frac=float(result.getOutput("sat")),
                dark_frac=float(result.getOutput("dark")))
