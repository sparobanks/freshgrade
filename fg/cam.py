"""Camera and colour handling: make the picture the model sees consistent.

Two cameras, or the same camera on a different day, will not record the same
colours. Lighting changes too. If the model trains on one look and meets
another, it can grade on the lighting instead of the fruit. The KTP brief
lists image formation, lighting and camera calibration as needed knowledge,
and this is where that lives.

Two simple, standard steps run before every image reaches the model:

* grey-world white balance  assume the average colour of the scene is neutral
  grey, and scale each channel to make it so. This cancels a colour cast from
  an un-balanced camera.
* contrast stretch          spread the brightness to a standard range, so a
  dark or washed-out capture looks the same to the model.

Both are classic, explainable operations, not learned, so the model's input is
predictable. The effect is measured in the evaluation: the same model is tested
with and without this step on the unseen "line" camera.
"""
from __future__ import annotations

import numpy as np

MEAN = np.array([0.45, 0.42, 0.38], dtype=np.float32)
STD = np.array([0.24, 0.23, 0.22], dtype=np.float32)


def grey_world(img: np.ndarray) -> np.ndarray:
    """White-balance a float image (H,W,3) in [0,1] by the grey-world rule."""
    means = img.reshape(-1, 3).mean(0)
    grey = means.mean()
    gain = grey / np.clip(means, 1e-4, None)
    gain = np.clip(gain, 0.6, 1.6)          # do not over-correct
    return np.clip(img * gain[None, None, :], 0, 1)


def contrast_stretch(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Stretch brightness to a standard range using robust percentiles."""
    v = img.mean(-1)
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    if b - a < 1e-3:
        return img
    return np.clip((img - a) / (b - a), 0, 1)


def preprocess(img_uint8: np.ndarray, calibrate: bool = True) -> np.ndarray:
    """uint8 (H,W,3) -> float (3,H,W) ready for the model.

    With calibrate=False the colour steps are skipped, so the evaluation can
    show what they are worth on an unseen camera.
    """
    img = img_uint8.astype(np.float32) / 255.0
    if calibrate:
        img = grey_world(img)
        img = contrast_stretch(img)
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))


def batch(imgs_uint8: np.ndarray, calibrate: bool = True) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,3,H,W) float32."""
    out = np.stack([preprocess(im, calibrate) for im in imgs_uint8])
    return out.astype(np.float32)
