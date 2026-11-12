import numpy as np

from fg import cam


def test_preprocess_shape():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    x = cam.preprocess(img)
    assert x.shape == (3, 64, 64)
    assert x.dtype == np.float32


def test_grey_world_reduces_a_colour_cast():
    base = np.full((64, 64, 3), 0.5, np.float32)
    cast = base * np.array([1.3, 1.0, 0.7])        # a strong red cast
    fixed = cam.grey_world(np.clip(cast, 0, 1))
    spread_before = np.ptp(cast.reshape(-1, 3).mean(0))
    spread_after = np.ptp(fixed.reshape(-1, 3).mean(0))
    assert spread_after < spread_before


def test_batch_shape():
    imgs = (np.random.rand(5, 64, 64, 3) * 255).astype(np.uint8)
    assert cam.batch(imgs).shape == (5, 3, 64, 64)
