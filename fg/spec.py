"""The grading spec: how a piece of fruit is graded against a retailer standard.

A real packhouse grades to a written specification agreed with the retailer.
This module writes that specification down as plain rules, so that every
synthetic piece of fruit we render has a known, correct grade. The grade is
the *worst* category: one serious defect is enough to drop a fruit, which is
how real specifications work.

Four things are judged:

* surface defect  how much of the skin is bruised or blemished
* size            the diameter in millimetres, against a target range
* colour          how far the skin colour is from ripe and on-spec
* shape           how far from round (deformed fruit is downgraded)

Keeping the spec in one place means the model is never graded against a moving
target, and the thresholds can be changed for a different product or retailer
without touching anything else.
"""
from __future__ import annotations

from dataclasses import dataclass

# The three grades, best to worst. Index order matters: higher = worse.
GRADES = ["Class I", "Class II", "Reject"]
CLASS_I, CLASS_II, REJECT = 0, 1, 2


@dataclass(frozen=True)
class Spec:
    # surface defect as a fraction of the visible skin
    defect_classII: float = 0.03      # above this, no longer Class I
    defect_reject: float = 0.10       # above this, rejected

    # size (diameter) in millimetres
    size_target: tuple[float, float] = (60.0, 80.0)   # ideal Class I band
    size_classII: tuple[float, float] = (55.0, 85.0)  # still sellable
    size_reject: tuple[float, float] = (50.0, 90.0)   # outside this, rejected

    # colour: distance from the ripe on-spec colour (0 = perfect)
    colour_classII: float = 0.18
    colour_reject: float = 0.34

    # shape: 0 = perfectly round, higher = more deformed
    shape_classII: float = 0.20
    shape_reject: float = 0.38


DEFAULT = Spec()


def _band_grade(value: float, inner: tuple[float, float], outer: tuple[float, float]) -> int:
    """Grade a value that has an ideal band (inner) and a sellable band (outer)."""
    lo_i, hi_i = inner
    lo_o, hi_o = outer
    if value < lo_o or value > hi_o:
        return REJECT
    if value < lo_i or value > hi_i:
        return CLASS_II
    return CLASS_I


def _threshold_grade(value: float, classII: float, reject: float) -> int:
    """Grade a value where lower is better (defect, colour distance, shape)."""
    if value >= reject:
        return REJECT
    if value >= classII:
        return CLASS_II
    return CLASS_I


def grade(defect_fraction: float, size_mm: float, colour_dist: float,
          shape_dev: float, spec: Spec = DEFAULT) -> int:
    """Return the overall grade (0/1/2) for one fruit, worst category wins."""
    parts = [
        _threshold_grade(defect_fraction, spec.defect_classII, spec.defect_reject),
        _band_grade(size_mm, spec.size_target, spec.size_reject),
        _threshold_grade(colour_dist, spec.colour_classII, spec.colour_reject),
        _threshold_grade(shape_dev, spec.shape_classII, spec.shape_reject),
    ]
    return max(parts)


def reasons(defect_fraction: float, size_mm: float, colour_dist: float,
            shape_dev: float, spec: Spec = DEFAULT) -> list[str]:
    """Plain-English reason(s) the fruit is not Class I, worst first.

    This is the *true* reason from the specification, used to check that the
    model's own explanation points at the right thing.
    """
    out: list[tuple[int, str]] = []
    g = _threshold_grade(defect_fraction, spec.defect_classII, spec.defect_reject)
    if g > CLASS_I:
        out.append((g, f"surface defect covers {defect_fraction*100:.0f}% of the skin"))
    g = _band_grade(size_mm, spec.size_target, spec.size_reject)
    if g > CLASS_I:
        side = "small" if size_mm < spec.size_target[0] else "large"
        out.append((g, f"{side} for the size band ({size_mm:.0f} mm)"))
    g = _threshold_grade(colour_dist, spec.colour_classII, spec.colour_reject)
    if g > CLASS_I:
        out.append((g, "skin colour is off-spec"))
    g = _threshold_grade(shape_dev, spec.shape_classII, spec.shape_reject)
    if g > CLASS_I:
        out.append((g, "shape is misshapen"))
    out.sort(key=lambda t: -t[0])
    return [r for _, r in out]


# the category a reason belongs to, used by the grounding check
def worst_category(defect_fraction: float, size_mm: float, colour_dist: float,
                   shape_dev: float, spec: Spec = DEFAULT) -> str:
    cats = {
        "defect": _threshold_grade(defect_fraction, spec.defect_classII, spec.defect_reject),
        "size": _band_grade(size_mm, spec.size_target, spec.size_reject),
        "colour": _threshold_grade(colour_dist, spec.colour_classII, spec.colour_reject),
        "shape": _threshold_grade(shape_dev, spec.shape_classII, spec.shape_reject),
    }
    return max(cats, key=lambda k: cats[k])
