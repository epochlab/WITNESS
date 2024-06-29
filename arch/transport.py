#!/usr/bin/env python3

import re
import numpy as np
from PIL import Image
import OpenEXR
import Imath

from arch import utils

def sort_key(item: list) -> list:
    name = item[0]
    if name == 'RGB': return (0, '')
    elif name == 'A': return (1, '')
    else: return (2, int(name.split('_')[0]))

def read(file: str) -> list:
    handle = OpenEXR.InputFile(file)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header = handle.header()
    dw = header['dataWindow']
    h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1

    layers = []
    rgb_layers = ['R', 'G', 'B']
    RGB = np.stack([np.frombuffer(handle.channel(c, pt), dtype=np.float32).reshape((h, w)) for c in rgb_layers], axis=-1)
    layers.append(('RGB', utils.array2image(np.power(RGB, 1/2.2))))

    channels = header['channels']
    pattern = re.compile(r"([^.]*)\.")
    single_channels = {ch for ch in channels if not pattern.match(ch) and ch not in rgb_layers}
    for name in single_channels:
        data = np.frombuffer(handle.channel(f'{name}', pt), dtype=np.float32).reshape((h, w))
        layers.append((f'{name}', utils.array2image(np.power(data, 1/2.2))))

    prefix_channels = {pattern.match(ch).group(1) for ch in channels if pattern.match(ch)}
    for name in prefix_channels:
        data = np.stack([np.frombuffer(handle.channel(f'{name}.{c}', pt), dtype=np.float32).reshape((h, w)) for c in ['X', 'Y', 'Z']], axis=-1)
        layers.append((f'{name}', utils.array2image(np.power(data, 1/2.2))))

    return sorted(layers, key=sort_key)

def write(rgb: np.ndarray, aov: dict[str, Image.Image], filename: str) -> OpenEXR.Header:
    header = OpenEXR.Header(rgb.shape[1], rgb.shape[0])
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

    rgb = np.power(rgb, 2.2)
    channels = {'R': rgb[:,:,0].tobytes(),
                'G': rgb[:,:,1].tobytes(),
                'B': rgb[:,:,2].tobytes(),
                }

    aov = {f'{idx}_{key}': value for idx, (key, value) in enumerate(aov.items())}
    for k, v in aov.items():
        v = np.power(v, 2.2).astype(np.float32)
        if v.ndim == 3 and v.shape[2] == 3:
            channels.update({f'{k}.{chan}': v[:, :, idx].tobytes() for idx, chan in enumerate(['X', 'Y', 'Z'])})
        elif v.ndim == 2:
            channels[f'{k}'] = v.tobytes()

    header['channels'] = {name: float_chan for name in channels.keys()}
    exr_file = OpenEXR.OutputFile(filename, header)
    exr_file.writePixels(channels)
    exr_file.close()
    return header