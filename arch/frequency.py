#!/usr/bin/env python3

from typing import Tuple
import numpy as np
import cv2 as cv

from arch import utils, filter

def calc_fft(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = filter.grayscale(img)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    phase = np.angle(fshift)
    return mag, phase, fshift

def mask_fft(fft: list, radius: int, blur: int) -> Tuple[np.ndarray, np.ndarray]:
    r, c = fft.shape
    half = np.sqrt(r**2 + c**2) / 2
    k_size = int(2 * half * blur / 100) + 1
    rad = int(half * radius / 100)
    mask = np.zeros((r, c), dtype=np.float32)
    mask = cv.circle(mask, (c//2, r//2), rad, 1, cv.FILLED)
    mask = cv.GaussianBlur(mask, (k_size, k_size), 0)
    mask /= np.max(mask)
    return mask

def decode_low(fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f_ifftshift = np.fft.ifftshift(fft * mask)
    low = np.fft.ifft2(f_ifftshift) * np.prod(f_ifftshift.shape)
    return utils.minmax_norm(np.real(low))

def decode_high(fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f_ifftshift = np.fft.ifftshift(fft * (1-mask))
    high = np.fft.ifft2(f_ifftshift) * np.prod(f_ifftshift.shape)
    return utils.minmax_norm(np.abs(high))