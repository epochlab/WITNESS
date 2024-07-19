#!/usr/bin/env python3

import numpy as np

def psnr(I, K):
    mse = np.mean((I - K) ** 2)
    return 20 * np.log10(np.max(I) / np.sqrt(mse))