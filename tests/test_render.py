import numpy as np

from fg import spec
from fg.render import dataset, render_fruit, SIZE


def test_dataset_shapes():
    d = dataset(40, "train", 0)
    assert d["images"].shape == (40, SIZE, SIZE, 3)
    assert d["images"].dtype == np.uint8
    assert set(np.unique(d["labels"])).issubset({0, 1, 2})


def test_defect_mask_is_inside_the_fruit():
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = render_fruit(rng, "train")
        # every defect pixel must be on the fruit, never on the belt
        assert np.all(s["fruit_mask"][s["mask"]])


def test_label_matches_the_spec_recomputed():
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = render_fruit(rng, "train")
        g = spec.grade(s["defect_fraction"], s["size_mm"], s["colour_dist"], s["shape_dev"])
        assert g == s["grade"]


def test_most_fruit_pass():
    d = dataset(1000, "train", 7)
    share_class_I = (d["labels"] == 0).mean()
    assert 0.4 < share_class_I < 0.7       # a realistic line passes most fruit


def test_domains_differ():
    rng = np.random.default_rng(0)
    a = render_fruit(np.random.default_rng(0), "train")["image"]
    b = render_fruit(np.random.default_rng(0), "line")["image"]
    assert not np.array_equal(a, b)        # same seed, different camera look
