"""Augmentation: the single most important reason synthetic training transfers.

A model trained on one clean rendering learns that exact look, lighting and
camera, and then fails the moment a real camera is slightly different. The fix
is not a bigger model, it is to vary the training images so widely that the
real camera looks like just one more variation the model has already seen. This
is domain randomisation, and it is what lets a model trained on synthetic data
alone work on a different capture.

Each training image is copied a few times, and every copy gets a random mix of:

* brightness and gamma      a lighter or darker capture
* per-channel colour gain   an un-white-balanced camera (a colour cast)
* gaussian noise            sensor noise on a cheap or fast camera
* a soft blur               slight focus or motion
* a horizontal flip         fruit can sit either way on the belt

The evaluation reports the unseen-camera accuracy with and without this, so its
value is measured, not assumed.
"""
from __future__ import annotations

import numpy as np


def _augment_one(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    # brightness and gamma
    x = x * rng.uniform(0.65, 1.35)
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
    # directional light: a brightness ramp across a random direction, so the
    # model sees many light positions, not just the one it trained on
    if rng.random() < 0.85:
        ang = rng.uniform(0, 2 * np.pi)
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx = (xx / (w - 1)) * 2 - 1
        yy = (yy / (h - 1)) * 2 - 1
        ramp = 1.0 + rng.uniform(0.0, 0.38) * (np.cos(ang) * xx + np.sin(ang) * yy)
        x = x * ramp[:, :, None]
    # per-channel colour cast
    x = x * rng.uniform(0.80, 1.20, size=3)[None, None, :]
    # soft blur, sometimes
    if rng.random() < 0.4:
        x = (x + np.roll(x, 1, 0) + np.roll(x, -1, 0)
             + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5.0
    # noise
    x = x + rng.normal(0, rng.uniform(0.0, 0.06), x.shape)
    # horizontal flip
    if rng.random() < 0.5:
        x = x[:, ::-1, :]
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def augment_images(imgs: np.ndarray, labels: np.ndarray, copies: int = 2,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return copies*N augmented images and their (repeated) labels."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(imgs) * copies,) + imgs.shape[1:], dtype=np.uint8)
    y = np.zeros(len(imgs) * copies, dtype=labels.dtype)
    k = 0
    for c in range(copies):
        for i in range(len(imgs)):
            out[k] = _augment_one(imgs[i], rng)
            y[k] = labels[i]
            k += 1
    return out, y
