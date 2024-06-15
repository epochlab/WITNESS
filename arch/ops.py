#!/usr/bin/env python3

import numpy as np

def diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a * 255.0).astype(int) - (b * 255.0).astype(int)) / 255.0