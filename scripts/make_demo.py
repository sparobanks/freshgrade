"""Make the annotated demo panel for the README.

Renders a row of fruit, grades each with the trained model, and draws the grade,
confidence and the Grad-CAM heat map beside each one, so a reader can see at a
glance what the model does and that its attention lands on the real faults.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from fg import cam, sort
from fg.explain import grad_cam
from fg.model import load as load_model
from fg.render import render_fruit
from fg.spec import GRADES


def overlay(img, cmap):
    base = img.astype(np.float32)
    heat = np.stack([cmap, cmap * 0.1, cmap * 0.1], -1) * 255
    w = cmap[..., None] * 0.55
    return np.clip(base * (1 - w) + heat * w, 0, 255).astype(np.uint8)


def main():
    m = load_model(ROOT / "models" / "grader.pt")
    rng = np.random.default_rng(11)
    # pick a spread of outcomes to show
    picks = []
    want = {0: 2, 1: 2, 2: 2}
    while picks == [] or sum(want.values()) > 0:
        dom = "line" if rng.random() < 0.4 else "train"
        s = render_fruit(rng, dom)
        g = s["grade"]
        if want.get(g, 0) > 0:
            picks.append(s)
            want[g] -= 1
        if len(picks) > 40:
            break

    scale = 3
    cell = 64 * scale
    pad = 26
    cols = len(picks)
    W = cols * (cell + 10) + 10
    H = 2 * cell + 3 * pad
    canvas = Image.new("RGB", (W, H), (248, 249, 247))
    d = ImageDraw.Draw(canvas)
    d.text((12, 6), "Fruit  (top) and what the model looked at  (bottom)", fill=(40, 40, 40))

    for i, s in enumerate(picks):
        x = cam.preprocess(s["image"])
        cmap, cls, probs = grad_cam(m, x)
        img_big = Image.fromarray(s["image"]).resize((cell, cell), Image.NEAREST)
        ov_big = Image.fromarray(overlay(s["image"], cmap)).resize((cell, cell), Image.NEAREST)
        x0 = 10 + i * (cell + 10)
        canvas.paste(img_big, (x0, pad))
        canvas.paste(ov_big, (x0, 2 * pad + cell))
        conf = probs[cls]
        d.text((x0, pad + cell + 3), f"{GRADES[cls]}  {conf:.0%}", fill=(20, 20, 20))

    out = ROOT / "docs" / "img" / "demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
