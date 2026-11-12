"""Make synthetic fruit images with their grade known for free.

The KTP brief asks for a photorealistic rendering pipeline that produces
training images with automated annotation, and for a model trained on
synthetic data alone. This module is that pipeline, kept deliberately simple
so it runs anywhere.

Each fruit is drawn as a shaded sphere (a simple lighting model: diffuse plus
a small highlight), given a size, a skin colour, surface defects and a shape,
on a conveyor-belt background. Because we choose those numbers, we know the
true defect area, size, colour and shape, so the grade (from `spec.py`) and a
pixel-accurate defect mask come out automatically. No hand labelling.

Two "domains" are provided:

* ``train``   even lighting, plain belt, light noise. What the model learns on.
* ``line``    a different light direction and strength, a textured belt, a
              colour cast (as a different camera would give), blur and noise.
              The model never trains on this, so testing on it shows whether
              the model learned the fruit or just the studio.

The colour cast in the ``line`` domain is why the model needs the colour
normalisation step in ``cam.py``: it stands in for a camera that is not
white-balanced the same way as the one used for training.
"""
from __future__ import annotations

import numpy as np

from . import spec

SIZE = 64                 # image is SIZE x SIZE pixels
MM_PER_PX = 1.46          # maps fruit radius in pixels to millimetres
RIPE = np.array([0.93, 0.55, 0.12])     # on-spec ripe skin colour (orange)
_GREEN = np.array([0.42, 0.58, 0.18])   # unripe
_PALE = np.array([0.95, 0.86, 0.55])    # washed out
BELT = np.array([0.30, 0.31, 0.33])     # conveyor grey


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def render_fruit(rng: np.random.Generator, domain: str = "train") -> dict:
    """Render one fruit. Returns image, defect mask and the true parameters.

    Domains: ``train`` (clean studio), ``line`` (a different, un-balanced
    camera and lighting), ``line_poor`` (that camera on a bad day: heavier
    noise and softer focus, standing in for a cheap or fast capture).
    """
    poor = domain == "line_poor"
    rand = domain == "rand"
    line = domain in ("line", "line_poor")

    # --- choose the true parameters -------------------------------------
    # Most fruit on a real line pass, so sample so that most are good and a
    # minority have a real problem. Geometry is fixed first, then the grade is
    # read off the geometry, so the label always matches the pixels.
    if rng.random() < 0.80:
        want_mm = rng.normal(70.0, 5.0)                # in the ideal size band
    else:
        want_mm = rng.uniform(50.0, 90.0)              # the off-size minority
    ecc = float(rng.normal(0.0, 0.09))                 # shape: 0 = round
    if rng.random() < 0.12:
        ecc += float(rng.normal(0.0, 0.22))            # a tail of odd shapes

    rm = want_mm / ((2.0 + ecc) * MM_PER_PX)           # mean radius in pixels
    rm = float(np.clip(rm, 15.0, 29.0))
    rx, ry = rm, rm * (1.0 + ecc)
    ry = float(np.clip(ry, 12.0, 30.0))
    shape_dev = abs(rx - ry) / ((rx + ry) / 2.0)
    size_mm = (rx + ry) * MM_PER_PX                    # diameter in mm (final)

    colour_dist = float(abs(rng.normal(0.0, 0.07)))    # most fruit on-colour
    if rng.random() < 0.12:
        colour_dist += float(rng.uniform(0.0, 0.40))   # the off-colour minority
    off = _GREEN if rng.random() < 0.6 else _PALE
    a = min(colour_dist, 0.6)
    base = (1 - a) * RIPE + a * off

    n_blobs = 0 if rng.random() < 0.68 else int(rng.poisson(1.2))
    if rng.random() < 0.09:
        n_blobs += int(rng.integers(2, 5))             # a minority are bruised

    # --- geometry and lighting ------------------------------------------
    cy = SIZE / 2 + rng.uniform(-4, 4)
    cx = SIZE / 2 + rng.uniform(-4, 4)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    nx = (xs - cx) / rx
    ny = (ys - cy) / ry
    rr = nx * nx + ny * ny
    inside = rr <= 1.0
    nz = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))          # sphere front surface
    normal = np.stack([nx, ny, nz], axis=-1)

    # Lighting. The training domain ("rand") randomises the light direction and
    # strength widely, so the model sees fruit lit from every side. Each test
    # camera ("train", "line", "line_poor") uses one fixed light held out from
    # training, so testing shows whether the model learned the fruit or the one
    # lighting it grew up with.
    if rand:
        light = _unit(np.array([rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9),
                                rng.uniform(0.5, 1.0)]))
        ambient = float(rng.uniform(0.25, 0.42))
    elif domain == "line":
        light = _unit(np.array([0.6, -0.3, 0.8]))
        ambient = 0.28
    elif poor:
        light = _unit(np.array([0.4, 0.5, 0.8]))
        ambient = 0.30
    else:
        light = _unit(np.array([-0.5, -0.6, 0.9]))
        ambient = 0.35
    diffuse = np.clip((normal * light).sum(-1), 0.0, 1.0)
    shade = ambient + (1 - ambient) * diffuse
    # small specular highlight
    spec_term = np.clip((normal * light).sum(-1), 0, 1) ** 24
    shade = np.clip(shade + 0.5 * spec_term, 0, 1.3)

    img = BELT.copy()[None, None, :] * np.ones((SIZE, SIZE, 1))
    skin = base[None, None, :] * shade[..., None]
    img = np.where(inside[..., None], skin, img)

    # --- surface defects (bruises) --------------------------------------
    defect = np.zeros((SIZE, SIZE), dtype=float)
    for _ in range(n_blobs):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, 0.85)
        bx = cx + np.cos(ang) * rad * rx
        by = cy + np.sin(ang) * rad * ry
        br = rng.uniform(2.5, 6.0)
        d2 = (xs - bx) ** 2 + (ys - by) ** 2
        blob = np.exp(-d2 / (2 * br * br))
        defect += blob
    defect = defect * inside
    defect_mask = defect > 0.45
    # paint bruises as dark brown
    brown = np.array([0.33, 0.19, 0.09])
    w = np.clip(defect, 0, 1)[..., None]
    img = img * (1 - w) + brown[None, None, :] * shade[..., None] * w
    frac = float(defect_mask.sum()) / max(1, int(inside.sum()))

    # --- camera / domain effects ----------------------------------------
    if rand:
        # randomise the camera too: cast, focus and noise all vary widely
        img = img * rng.uniform(0.80, 1.20, size=3)[None, None, :]
        b = int(rng.integers(0, 3))
        if b:
            img = _blur(img, b)
        noise = float(rng.uniform(0.01, 0.09))
    elif line:
        cast = rng.uniform(0.85, 1.15, size=3)          # un-white-balanced camera
        img = img * cast[None, None, :]
        img = _blur(img, 2 if poor else 1)              # soft focus / upscaled
        noise = 0.09 if poor else 0.05
    else:
        noise = 0.02
    img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 1)

    g = spec.grade(frac, size_mm, colour_dist, shape_dev)
    return {
        "image": (img * 255).astype(np.uint8),
        "mask": defect_mask,
        "fruit_mask": inside,
        "defect_fraction": frac,
        "size_mm": size_mm,
        "colour_dist": colour_dist,
        "shape_dev": shape_dev,
        "grade": g,
        "domain": domain,
    }


def _blur(img: np.ndarray, k: int) -> np.ndarray:
    """Tiny box blur, so we do not depend on scipy/opencv for this."""
    out = img.copy()
    for _ in range(k):
        out = (out
               + np.roll(out, 1, 0) + np.roll(out, -1, 0)
               + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5.0
    return out


def dataset(n: int, domain: str, seed: int) -> dict:
    """Render n fruit. Returns images (n,64,64,3) uint8, labels, masks, params."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, SIZE, SIZE, 3), dtype=np.uint8)
    masks = np.zeros((n, SIZE, SIZE), dtype=bool)
    fruit = np.zeros((n, SIZE, SIZE), dtype=bool)
    y = np.zeros(n, dtype=np.int64)
    params = []
    for i in range(n):
        s = render_fruit(rng, domain)
        imgs[i] = s["image"]
        masks[i] = s["mask"]
        fruit[i] = s["fruit_mask"]
        y[i] = s["grade"]
        params.append({k: s[k] for k in
                       ("defect_fraction", "size_mm", "colour_dist", "shape_dev", "grade")})
    return {"images": imgs, "labels": y, "masks": masks, "fruit_mask": fruit,
            "params": params, "domain": domain}
