"""Show why the model gave a grade, and check it is looking at the real defect.

A grader that only outputs a label is hard to trust on a line. Staff need to see
what the model reacted to. Grad-CAM (Selvaraju et al., 2017) turns the last
convolution layer into a heat map over the fruit, showing the area that pushed
the grade.

On synthetic data we also know the true defect mask, so we can check something
a label alone cannot tell us: when the model downgrades a fruit for a surface
defect, does its heat map actually land on the bruise, or did it get the right
answer for the wrong reason? That check is reported in the evaluation. In
production the mask is not available, so the heat map is for the operator and
the grounding number is the evidence, measured offline, that it can be trusted.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import cam
from .spec import GRADES


def grad_cam(model, x_np: np.ndarray, target: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """Heat map (64,64) in [0,1] for one preprocessed image x_np (3,64,64)."""
    model.eval()
    x = torch.tensor(x_np[None], dtype=torch.float32, requires_grad=True)

    acts, grads = {}, {}
    h1 = model.last_conv.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = model.last_conv.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    logits = model(x)
    probs = F.softmax(logits / model.temperature, dim=1)[0].detach().numpy()
    cls = int(probs.argmax()) if target is None else target
    model.zero_grad()
    logits[0, cls].backward()
    h1.remove(); h2.remove()

    a = acts["v"][0]                      # (C,h,w)
    g = grads["v"][0]                     # (C,h,w)
    weights = g.mean(dim=(1, 2))          # importance of each channel
    cam_map = F.relu((weights[:, None, None] * a).sum(0))
    cam_map = cam_map.detach().numpy()
    if cam_map.max() > 0:
        cam_map = cam_map / cam_map.max()
    cam_map = _resize(cam_map, 64, 64)
    return cam_map, cls, probs


def _resize(a: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbour upsample, no external dependency."""
    yi = (np.linspace(0, a.shape[0] - 1, H)).round().astype(int)
    xi = (np.linspace(0, a.shape[1] - 1, W)).round().astype(int)
    return a[yi][:, xi]


def grounding(cam_map: np.ndarray, defect_mask: np.ndarray) -> float:
    """Share of the heat map's weight that sits on the true defect.

    Kept for completeness, but on its own it reads low because a bruise is a
    tiny part of the fruit. The ratio and pointing-game measures below are the
    fair ones. Only meaningful when the fruit was downgraded for a defect.
    """
    total = cam_map.sum()
    if total <= 0 or defect_mask.sum() == 0:
        return 0.0
    return float((cam_map * defect_mask).sum() / total)


def grounding_ratio(cam_map: np.ndarray, defect_mask: np.ndarray,
                    fruit_mask: np.ndarray) -> float:
    """How much more attention the heat map puts on the bruise than elsewhere.

    Mean heat on the real defect divided by mean heat on the rest of the fruit.
    1.0 means no preference; above 1.0 means the model is looking more at the
    real bruise, which is what we want.
    """
    rest = fruit_mask & ~defect_mask
    if defect_mask.sum() == 0 or rest.sum() == 0:
        return 0.0
    on = cam_map[defect_mask].mean()
    off = cam_map[rest].mean()
    return float((on + 1e-9) / (off + 1e-9))


def pointing_hit(cam_map: np.ndarray, defect_mask: np.ndarray, radius: int = 3) -> bool:
    """Does the hottest point of the heat map land on (or next to) a bruise?

    The pointing game, a standard weak-localisation check: find where the heat
    map peaks and see if a real defect pixel is within `radius` of it.
    """
    if defect_mask.sum() == 0:
        return False
    y, x = np.unravel_index(int(np.argmax(cam_map)), cam_map.shape)
    H, W = defect_mask.shape
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)
    return bool(defect_mask[y0:y1, x0:x1].any())


def band(grade: int) -> str:
    return GRADES[grade]


def note(grade: int, confidence: float) -> str:
    """One plain line for the operator."""
    name = GRADES[grade]
    if grade == 0:
        return f"Pass as {name} (confidence {confidence:.0%}). No problem found on the skin."
    if grade == 1:
        return (f"Down to {name} (confidence {confidence:.0%}). A fault was found; "
                f"the highlighted area shows where.")
    return (f"Reject (confidence {confidence:.0%}). A clear fault was found; "
            f"the highlighted area shows where.")


def explain_image(model, img_uint8: np.ndarray, calibrate: bool = True) -> dict:
    """Full explanation for one raw image: grade, confidence, heat map, note."""
    x = cam.preprocess(img_uint8, calibrate=calibrate)
    cam_map, cls, probs = grad_cam(model, x)
    return {
        "grade": cls,
        "grade_name": GRADES[cls],
        "confidence": float(probs[cls]),
        "probs": probs.tolist(),
        "cam": cam_map,
        "note": note(cls, float(probs[cls])),
    }
