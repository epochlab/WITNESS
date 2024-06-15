#!/usr/bin/env python3

from typing import Tuple, List

import dlib
import cv2 as cv
import numpy as np

MODEL_PATH = '../models/shape_predictor_68_face_landmarks.dat'

predictor = dlib.shape_predictor(MODEL_PATH)
detector = dlib.get_frontal_face_detector()

def detect_bounds(img: np.ndarray) -> List:
    return detector(img)

def extract_features(img: np.ndarray, bbox: List):
    shapes = []
    for b in bbox:
        shape = predictor(img, b)
        for part in shape.parts():
            shapes.append(part)
    return shapes

def visualize(img: np.ndarray, bbox: list=None, points: Tuple=None) -> np.ndarray:
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if bbox:
        for b in bbox:
            x1, y1, x2, y2 = b.left(), b.top(), b.right(), b.bottom()
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
    if points:
        for p in points:
            cv.circle(img, (p.x, p.y), 2, (255, 255, 255), -1)
    return img