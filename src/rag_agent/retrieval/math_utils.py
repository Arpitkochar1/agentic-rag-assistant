from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
