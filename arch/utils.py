#!/usr/bin/env python3

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def device_mapper() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def load_image(file: str) -> Image.Image:
    return Image.open(file)

def image2array(x, dtype='float32'):
    return np.array(x, dtype=dtype) / 255.0

def array2image(x: np.ndarray) -> Image.Image:
    return Image.fromarray((x * 255.0).astype("uint8"))

def minmax_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def x_norm(img):
    n_channels = [minmax_norm(img[:,:,i]) for i in range(img.shape[2])]
    return np.stack(n_channels, axis=2)

def create_lut(low: float, high: float) -> np.ndarray:
    p1 = (max(0, low), max(0, -low))
    p2 = (min(1, 1 - high), min(1, 1 + high))

    if p1[0] == p2[0]: return np.full(256, 1.0, np.float32)

    x = np.linspace(0, 1, 256)
    lut = (x * (p1[1] - p2[1]) + p1[0] * p2[1] - p1[1] * p2[0]) / (p1[0] - p2[0])
    return np.clip(lut, 0, 1).astype(np.float32)

def contact_layer(imgs: list, rows: int, cols: int, labels: list=None, font_size: int=20, offset: int=10) -> Image.Image:
    if imgs[0].ndim == 2: h, w = imgs[0].shape
    elif imgs[0].ndim == 3: h, w, _ = imgs[0].shape
    grid = Image.new('RGB', size=(cols*w, rows*h))
    font = ImageFont.load_default(font_size)
    for i, img in enumerate(imgs):
        img = array2image(img)
        if labels:
            draw = ImageDraw.Draw(img)
            text = labels[i]
            draw.text((offset, offset), text, font=font, fill="#FFF")
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid