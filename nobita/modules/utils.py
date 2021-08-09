from typing import Tuple, List
import numpy as np
import cv2


def frame_norm(frame: np.ndarray, bbox:np.ndarray):
    return (
        np.clip(np.array(bbox), 0, 1)
        * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]
    ).astype(int)


def to_planar(arr: np.ndarray, shape: Tuple[int, int]) -> List:
    return [
        val
        for channel in cv2.resize(arr, shape).transpose(2, 0, 1)
        for y_col in channel
        for val in y_col
    ]
