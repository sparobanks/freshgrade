"""Turn a grade into a sorting decision, and put a money value on getting it right.

On the line, a grade has to become an action: send the fruit to the premium
pack, the second-grade pack (juice or discount), or off the line. This module
makes that decision, and it also answers the question a business actually asks:
what is better grading worth?

The value model is a simple payoff table. A fruit sold in a lane earns that
lane's price. Selling a fruit as better than it is (a bruised fruit in the
premium pack) carries a penalty, because it comes back as a complaint or a
rejected pallet. Selling a good fruit as second-grade just loses the
difference. The numbers are illustrative, not Karsten's, and live in one place
so they are easy to change.

The headline number compares the vision grader against a size-only grader, the
kind of mechanical sizer a low-tech line uses. The size grader cannot see
bruises, off-colour or odd shapes, so the gap between them is, in plain terms,
the value of being able to see the fruit.
"""
from __future__ import annotations

import numpy as np

from . import spec

# lane prices (pounds per fruit) and the review lane
PRICE = {0: 0.30, 1: 0.15, 2: 0.00}       # Class I, Class II, Reject

# payoff[assigned_lane][true_grade]: what we actually earn
# overselling (lane better than the truth) is penalised; underselling loses value
PAYOFF = np.array([
    [0.30, -0.10, -0.50],   # assigned Class I
    [0.15,  0.15, -0.20],   # assigned Class II
    [0.00,  0.00,  0.00],   # assigned Reject
])

REVIEW_CONF = 0.55          # below this confidence, send for a human look


def decide(grade: int, confidence: float) -> dict:
    """Lane and action for one fruit."""
    if confidence < REVIEW_CONF:
        return {"lane": "review", "action": "send for a quick human check"}
    lane = ["premium pack", "second-grade pack", "off the line"][grade]
    action = ["pack as Class I", "pack as Class II", "divert (reject)"][grade]
    return {"lane": lane, "action": action}


def size_only_grade(size_mm: float) -> int:
    """A mechanical sizer: grades on diameter alone, blind to everything else."""
    from .spec import _band_grade, DEFAULT
    return _band_grade(size_mm, DEFAULT.size_target, DEFAULT.size_reject)


def value(assigned: np.ndarray, true_grade: np.ndarray) -> float:
    """Total pounds earned for a batch, given lane assignments and true grades."""
    return float(PAYOFF[assigned, true_grade].sum())


def value_report(y_true: np.ndarray, y_model: np.ndarray, params: list[dict],
                 per: int = 1000) -> dict:
    """Compare the vision grader with a perfect grader and a size-only grader.

    Returns pounds per `per` fruit for each, and the model's share of the best
    possible value (the "value recovery").
    """
    n = len(y_true)
    size_grade = np.array([size_only_grade(p["size_mm"]) for p in params])
    pass_all = np.zeros(n, dtype=int)                    # no grading: all premium

    v_perfect = value(y_true, y_true)                    # grade every fruit correctly
    v_model = value(y_model, y_true)
    v_size = value(size_grade, y_true)
    v_pass = value(pass_all, y_true)

    floor = v_pass                                       # grading nothing
    span = max(1e-6, v_perfect - floor)
    scale = per / n
    return {
        "fruit": n,
        "per": per,
        "perfect_grader": round(v_perfect * scale, 2),
        "vision_grader": round(v_model * scale, 2),
        "size_only_grader": round(v_size * scale, 2),
        "no_grading": round(v_pass * scale, 2),
        "value_recovered_vs_size_only": round((v_model - v_size) * scale, 2),
        "value_recovered_vs_no_grading": round((v_model - v_pass) * scale, 2),
        "share_of_best_possible": round((v_model - floor) / span, 3),
        "size_only_share_of_best": round((v_size - floor) / span, 3),
    }
