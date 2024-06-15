#!/usr/bin/env python3

import io
from PIL import Image
import numpy as np

from arch import utils

def error_level_analysis(file: str, quality: float=95) -> np.ndarray:
    with Image.open(file) as x:
        buffer = io.BytesIO()
        x.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        x0 = np.array(x, dtype=np.int8)
        x1 = np.array(Image.open(buffer), dtype=np.int8)
        return utils.minmax_norm(np.abs(x0 - x1))