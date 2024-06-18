#!/usr/bin/env python3

import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from arch import utils
    
def compress(x, quality):
    buffer = io.BytesIO()
    x.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def diff(a, b):
    return np.abs(np.array(a, dtype=np.int8) - np.array(b, dtype=np.int8))

def compute(x, quality):
    return utils.minmax_norm(diff(x, compress(x, int(quality))))

def loss_curve(x, qualities=tuple(range(1, 101))):
    loss = [np.mean(diff(x, compress(x, q))) for q in qualities]
    loss = utils.minmax_norm(loss)
    qm = np.argmin(loss) + 1
    return loss, qm

def plot_curve(loss, qm):
    nrange = np.arange(1, 101)
    plt.plot(nrange, loss,label="compression loss", lw=0.5)
    plt.fill_between(nrange, loss, alpha=0.1)
    plt.axvline(qm, linestyle=":", color="r", label=f"min error (q = {qm})")
    plt.xlim([1, 100]), plt.ylim([0, 1])
    plt.xlabel('Quality'), plt.ylabel('Compression Loss')
    plt.grid(alpha=0.25), plt.legend(loc="upper center")
    plt.tight_layout()