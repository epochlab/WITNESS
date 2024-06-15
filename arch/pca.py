#!/usr/bin/env python3

import numpy as np
import cv2 as cv

from arch import utils

def extract_components(img: np.ndarray):
    h, w, c = img.shape
    x = np.reshape(img, (h * w, c))

    mu, ev, _ = cv.PCACompute2(x, np.array([]))
    p = np.reshape(cv.PCAProject(x, mu, ev), (h, w, c))
    x0 = img - mu

    output = []
    for i, v in enumerate(ev):
        cross = np.cross(x0, v)
        distance = np.linalg.norm(cross, axis=2) / np.linalg.norm(v)
        project = p[:, :, i]
        output.extend([utils.minmax_norm(distance), utils.minmax_norm(project), utils.x_norm(cross)])
    return output