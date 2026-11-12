import numpy as np

from fg import cam, model
from fg.explain import grad_cam, grounding
from fg.model import GradeNet


def _toy(n=180, seed=0):
    """Three easy classes: a bright block in a different place for each class.

    The cue is spatial, not global brightness, so it survives the colour
    normalisation (which deliberately removes brightness).
    """
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 64, 64, 3), np.uint8)
    y = np.zeros(n, np.int64)
    boxes = [(5, 25, 5, 25), (22, 42, 22, 42), (38, 58, 38, 58)]   # TL, centre, BR
    for i in range(n):
        c = i % 3
        base = np.full((64, 64, 3), 90, np.float32) + rng.normal(0, 10, (64, 64, 3))
        y0, y1, x0, x1 = boxes[c]
        base[y0:y1, x0:x1] = 220
        imgs[i] = np.clip(base, 0, 255).astype(np.uint8)
        y[i] = c
    return cam.batch(imgs), y


def test_proba_is_a_distribution():
    m = GradeNet()
    X, _ = _toy(9)
    p = m.predict_proba(X)
    assert p.shape == (9, 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_model_learns_an_easy_task():
    Xtr, ytr = _toy(180, 0)
    Xva, yva = _toy(60, 1)
    m = model.train_model(Xtr, ytr, Xva, yva, epochs=6)
    acc = (m.predict_proba(Xva).argmax(1) == yva).mean()
    assert acc > 0.8


def test_grad_cam_shape_and_range():
    m = GradeNet()
    X, _ = _toy(3)
    cmap, cls, probs = grad_cam(m, X[0])
    assert cmap.shape == (64, 64)
    assert cmap.min() >= 0.0 and cmap.max() <= 1.0001
    assert 0 <= cls < 3


def test_grounding_rewards_overlap():
    mask = np.zeros((64, 64), bool)
    mask[20:30, 20:30] = True
    on = np.zeros((64, 64)); on[20:30, 20:30] = 1.0
    off = np.zeros((64, 64)); off[0:10, 0:10] = 1.0
    assert grounding(on, mask) > 0.9
    assert grounding(off, mask) < 0.1
