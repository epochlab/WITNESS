#!/usr/bin/env python3

import numpy as np
import cv2 as cv

from arch import utils, filter

def grayscale(x: np.ndarray) -> np.ndarray:
    return np.dot(x[...,:3], [0.2989, 0.5870, 0.1140])

def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    Xmax = np.amax(rgb, axis=2)
    Xmin = np.amin(rgb, axis=2)
    C = Xmax - Xmin
    C_safe = np.where(C==0, 1, C)

    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    hsv = np.zeros_like(rgb)
    hsv[..., 0] = np.where(Xmax==R, (G - B) / C_safe, 0)
    hsv[..., 0] = np.where(Xmax==G, ((B - R) / C_safe) + 2, hsv[..., 0])
    hsv[..., 0] = np.where(Xmax==B, ((R - G) / C_safe) + 4, hsv[..., 0])
    hsv[..., 0] = (hsv[..., 0] % 6) / 6
    hsv[..., 1] = np.where(Xmax == 0, 0, C_safe / Xmax)
    hsv[..., 2] = Xmax
    return hsv

def luminance_gradient(img: np.ndarray, intensity: float=95, mode: str='abs') -> np.ndarray:
    gray = filter.grayscale(img)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    pad_im = np.pad(gray, int(sobel_x.shape[0]//2), mode='reflect')

    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            patch = pad_im[i:i+sobel_x.shape[0], j:j+sobel_x.shape[1]]
            dx[i, j] = np.sum(patch * sobel_x)
            dy[i, j] = np.sum(patch * sobel_y)

    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)

    red = ((dx / np.max(dx_abs) * 0.5) + 0.5)
    green = ((dy / np.max(dy_abs) * 0.5) + 0.5)
    if mode == 'none': blue = np.zeros_like(red)
    elif mode == 'flat': blue = np.ones_like(red)
    elif mode == 'abs': blue = ((dx_abs + dy_abs) / np.max(dx_abs + dy_abs))
    elif mode == 'norm':
        magnitude = np.sqrt(dx**2 + dy**2)
        blue = (magnitude / np.max(magnitude))

    gradient_image = np.stack((red, green, blue), axis=-1)
    lut = utils.create_lut(intensity, intensity)
    return lut[(gradient_image * 255).astype(int)]

def echo_edge(img: np.ndarray, radius: int, contrast: float) -> np.ndarray:
    kernel_size = int(2 * radius + 1)
    lut = utils.create_lut(0, contrast)

    laplace = []
    for channel in cv.split(img):
        deriv = np.fabs(cv.Laplacian(channel, cv.CV_64F, None, kernel_size))
        deriv = utils.minmax_norm(deriv)
        laplace.append(lut[(deriv * 255).astype(int)])
    return np.stack(laplace, axis=-1)

def noise_seperation(img: np.ndarray, mode: str, radius: int, sigma: int) -> np.ndarray:
    kernel_size = radius * 2 + 1
    if mode == 'median': denoised = cv.medianBlur(img, kernel_size)
    elif mode == 'gaussian': denoised =  cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif mode == 'box': denoised = cv.blur(img, (kernel_size, kernel_size))
    elif mode == 'bilateral': denoised = cv.bilateralFilter(img, kernel_size, sigma, sigma)

    diff = np.abs(img - denoised)
    return utils.minmax_norm(diff)