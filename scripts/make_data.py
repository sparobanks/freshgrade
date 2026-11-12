"""Render the synthetic datasets once and cache them to disk.

`run_eval.py` renders on the fly, so this is optional. It is here so the data
can be inspected, reused, or rendered once and shared. Each file holds the
images, the true grades, the defect masks and the per-fruit parameters.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from fg.render import dataset

SETS = [("train", 8000, 0), ("val", 1500, 1), ("test", 2500, 2),
        ("line", 2500, 3), ("line_poor", 2000, 4)]


def main():
    (ROOT / "data").mkdir(exist_ok=True)
    for name, n, seed in SETS:
        domain = "train" if name in ("train", "val", "test") else name
        d = dataset(n, domain, seed)
        np.savez_compressed(ROOT / "data" / f"{name}.npz",
                            images=d["images"], labels=d["labels"], masks=d["masks"])
        print(f"{name:10s} {n:5d} fruit  ->  data/{name}.npz")


if __name__ == "__main__":
    main()
