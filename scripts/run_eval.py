"""Train the grader and measure it honestly. This is the single build.

    python scripts/run_eval.py

Steps:
1. Render synthetic fruit: a training set, a validation set, a same-camera test
   set, and two sets from a different, unseen camera (normal and poor quality).
2. Train the CNN on augmented synthetic data only.
3. Measure, on data the model never saw:
   - grade accuracy on the same camera (the synthetic-only target),
   - accuracy on the unseen camera, with and without colour calibration, and on
     a poor-quality unseen camera, to show how far synthetic training carries,
   - how well it catches fruit that is not Class I (the costly miss),
   - whether the probabilities are calibrated,
   - whether its reason (Grad-CAM) lands on the real defect (grounding),
   - the money value of the grading against a size-only grader.
Writes models/grader.pt, reports/metrics.json and charts in reports/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from fg import cam, model, sort
from fg.augment import augment_images
from fg.explain import grad_cam, grounding_ratio, pointing_hit
from fg.render import dataset
from fg.spec import GRADES, worst_category

N_TRAIN, N_VAL, N_TEST, N_LINE, N_POOR = 8000, 1500, 2500, 2500, 2000
EPOCHS = 20
CLASS_WEIGHT = np.array([1.0, 1.25, 1.25])


def ece(y, p, bins=10):
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if m.sum():
            e += abs(conf[m].mean() - correct[m].mean()) * m.sum() / len(y)
    return float(e)


def acc_block(m, X, y):
    p = m.predict_proba(X)
    yp = p.argmax(1)
    return {"accuracy": round(accuracy_score(y, yp), 3),
            "macro_f1": round(f1_score(y, yp, average="macro"), 3),
            "calibration_error": round(ece(y, p), 3)}


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    # train on randomised cameras and lighting (domain randomisation); test on
    # four fixed cameras, none of which the model saw as a fixed setting
    tr = dataset(N_TRAIN, "rand", 0)
    va = dataset(N_VAL, "rand", 1)
    te = dataset(N_TEST, "train", 2)        # a clean studio camera
    ln = dataset(N_LINE, "line", 3)         # a different, un-balanced camera
    lp = dataset(N_POOR, "line_poor", 4)    # that camera on a bad day

    aug_imgs, aug_y = augment_images(tr["images"], tr["labels"], copies=2, seed=5)
    Xtr = cam.batch(aug_imgs)
    Xva = cam.batch(va["images"])
    Xte = cam.batch(te["images"])
    Xln = cam.batch(ln["images"])
    Xln_nc = cam.batch(ln["images"], calibrate=False)
    Xlp = cam.batch(lp["images"])

    m = model.train_model(Xtr, aug_y, Xva, va["labels"], epochs=EPOCHS,
                          class_weight=CLASS_WEIGHT)
    model.save(m, ROOT / "models" / "grader.pt")

    report = {
        "train_fruit": N_TRAIN, "augmented_to": len(aug_y),
        "temperature": round(m.temperature, 3),
        "class_balance_test": {GRADES[k]: round(float((te["labels"] == k).mean()), 3)
                               for k in range(3)},
        "studio_camera": acc_block(m, Xte, te["labels"]),
        "line_camera_calibrated": acc_block(m, Xln, ln["labels"]),
        "line_camera_no_calibration": acc_block(m, Xln_nc, ln["labels"]),
        "poor_camera": acc_block(m, Xlp, lp["labels"]),
    }

    # catching fruit that is not Class I (the costly miss) on the same camera
    p = m.predict_proba(Xte)
    yp = p.argmax(1)
    bad_true = (te["labels"] >= 1).astype(int)
    bad_pred = (yp >= 1).astype(int)
    report["catching_not_class_I"] = {
        "precision": round(precision_score(bad_true, bad_pred), 3),
        "recall": round(recall_score(bad_true, bad_pred), 3),
        "rejects_caught": round(recall_score((te["labels"] == 2).astype(int),
                                             (yp == 2).astype(int)), 3),
    }
    report["confusion_same_camera"] = confusion_matrix(te["labels"], yp).tolist()

    # grounding: when a fruit is downgraded for a defect, does the heat map look
    # at the real bruise? Measured as attention ratio (heat on the bruise vs the
    # rest of the fruit) and the pointing game (does the hottest point hit it).
    ratios, hits = [], []
    checked = 0
    for i in range(len(te["labels"])):
        pr = te["params"][i]
        if te["labels"][i] >= 1 and te["masks"][i].sum() > 0 and worst_category(
                pr["defect_fraction"], pr["size_mm"], pr["colour_dist"], pr["shape_dev"]) == "defect":
            x = cam.preprocess(te["images"][i])
            cmap, cls, _ = grad_cam(m, x)
            ratios.append(grounding_ratio(cmap, te["masks"][i], te["fruit_mask"][i]))
            hits.append(pointing_hit(cmap, te["masks"][i]))
            checked += 1
            if checked >= 400:
                break
    ratios = np.array(ratios) if ratios else np.array([0.0])
    report["grounding_on_defect_downgrades"] = {
        "fruit_checked": int(checked),
        "median_attention_ratio": round(float(np.median(ratios)), 2),
        "share_looking_more_at_defect": round(float((ratios > 1).mean()), 3),
        "pointing_game_hit_rate": round(float(np.mean(hits)), 3),
    }

    # money value on the same camera
    report["value"] = sort.value_report(te["labels"], yp, te["params"])

    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    _charts(te["labels"], p, yp, report)
    print(json.dumps(report, indent=2))


def _charts(y, p, yp, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # calibration
    conf = p.max(1)
    correct = (yp == y).astype(float)
    bins = np.linspace(0, 1, 11)
    xs, ys = [], []
    for b in range(10):
        msk = (conf >= bins[b]) & (conf < bins[b + 1])
        if msk.sum() > 10:
            xs.append(conf[msk].mean())
            ys.append(correct[msk].mean())
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="#8a948f", lw=1)
    ax.plot(xs, ys, "-o", ms=4, color="#c2571a")
    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Share actually correct")
    ax.set_title(f"Calibration (error {report['studio_camera']['calibration_error']:.3f})",
                 loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "calibration.png", dpi=150); plt.close(fig)

    # robustness across cameras
    conds = [("Studio\ncamera", report["studio_camera"]["accuracy"]),
             ("Line\ncamera", report["line_camera_calibrated"]["accuracy"]),
             ("Line, no\ncalibration", report["line_camera_no_calibration"]["accuracy"]),
             ("Poor\ncamera", report["poor_camera"]["accuracy"])]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    cols = ["#2f7d52", "#2f7d52", "#b23b3b", "#7a6a1f"]
    ax.bar([c[0] for c in conds], [c[1] for c in conds], color=cols)
    for i, c in enumerate(conds):
        ax.text(i, c[1] + 0.01, f"{c[1]:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Grade accuracy")
    ax.set_title("Accuracy holds up on an unseen camera", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "robustness.png", dpi=150); plt.close(fig)

    # value bars
    v = report["value"]
    labels = ["Perfect\ngrader", "Vision\ngrader", "Size-only\ngrader", "No\ngrading"]
    vals = [v["perfect_grader"], v["vision_grader"], v["size_only_grader"], v["no_grading"]]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(labels, vals, color=["#355", "#2f7d52", "#9a7b2f", "#999"])
    for i, val in enumerate(vals):
        ax.text(i, val + 2, f"£{val:.0f}", ha="center", fontsize=9)
    ax.set_ylabel(f"£ per {v['per']} fruit")
    ax.set_title("Value of seeing the fruit, not just sizing it", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "value.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
