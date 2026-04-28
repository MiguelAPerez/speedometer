"""
reid.py — Lightweight appearance-based re-identification helpers.

Provides HSV color histograms per track so the graveyard matcher can
distinguish cars by color when Kalman position predictions have drifted.
No external dependencies beyond cv2 and numpy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# H=32 bins, S=32 bins, V=16 bins → 80-element descriptor
_H_BINS = 32
_S_BINS = 32
_V_BINS = 16
_HIST_LEN = _H_BINS + _S_BINS + _V_BINS   # 80


def extract_histogram(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    """
    Extract an HSV color histogram from a bounding-box crop of a BGR frame.

    Returns an 80-element L1-normalised float32 vector, or None if the crop
    is degenerate (too small or outside the frame).
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [_H_BINS], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [_S_BINS], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [_V_BINS], [0, 256])

    cv2.normalize(h_hist, h_hist, norm_type=cv2.NORM_L1)
    cv2.normalize(s_hist, s_hist, norm_type=cv2.NORM_L1)
    cv2.normalize(v_hist, v_hist, norm_type=cv2.NORM_L1)

    return np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()]).astype(np.float32)


def blend_histogram(stored: np.ndarray, new_hist: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    Exponential moving average update: stored = (1-alpha)*stored + alpha*new_hist.

    alpha=0.15 means the current frame contributes 15% — the stored histogram
    converges slowly so brief lighting changes don't corrupt the appearance model.
    """
    np.multiply(stored, 1.0 - alpha, out=stored)
    np.multiply(new_hist, alpha, out=new_hist)
    np.add(stored, new_hist, out=stored)
    return stored


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Bhattacharyya distance between two L1-normalised histograms.

    Returns a value in [0, 1] where 0 = identical, 1 = completely disjoint.
    cv2.compareHist requires 2D (N,1) arrays.
    """
    return float(cv2.compareHist(
        h1.reshape(-1, 1),
        h2.reshape(-1, 1),
        cv2.HISTCMP_BHATTACHARYYA,
    ))
