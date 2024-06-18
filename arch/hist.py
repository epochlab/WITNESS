#!/usr/bin/env python3

import numpy as np

def process(x):
    x = np.array(x)
    pixel_count = np.zeros([5, 256], dtype=int)

    for i in range(256):
        pixel_count[0,i] = i
        pixel_count[1,i] = np.count_nonzero(x == i)
        pixel_count[2,i] = np.count_nonzero(x[:,:,0] == i)
        pixel_count[3,i] = np.count_nonzero(x[:,:,1] == i)
        pixel_count[4,i] = np.count_nonzero(x[:,:,2] == i)

    pixels = x.shape[0] * x.shape[1]
    min = x.min()
    max = x.max()
    argmin = np.argmin(pixel_count[1])
    argmax = np.argmax(pixel_count[1])
    mean = round(np.mean(x), 2)
    std = round(np.std(x), 2)
    median = np.argmax(np.cumsum(pixel_count[1]) > np.sum(pixel_count[1]) / 2)
    nonzero = [np.nonzero(pixel_count[1])[0][0] + 0, np.nonzero(pixel_count[1])[0][-1]]
    empty_bins = np.sum(pixel_count[1] == 0)
    unique_colors = np.unique(np.reshape(x, (pixels, x.shape[2])), axis=0).shape[0]

    return pixel_count, (pixels, min, max, argmin, argmax, mean, std, median, nonzero, empty_bins, unique_colors)