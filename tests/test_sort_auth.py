import numpy as np

from fg import auth, sort


def test_decide_lanes():
    assert sort.decide(0, 0.9)["lane"] == "premium pack"
    assert sort.decide(2, 0.9)["lane"] == "off the line"
    assert sort.decide(0, 0.3)["lane"] == "review"      # low confidence


def test_size_only_grader_is_blind_to_defects():
    # a badly bruised but well-sized fruit looks fine to a size grader
    assert sort.size_only_grade(70.0) == 0


def test_value_report_keys_and_ordering():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    params = [{"size_mm": float(s)} for s in rng.uniform(50, 90, 400)]
    perfect = sort.value_report(y, y, params)        # model = truth
    assert perfect["vision_grader"] == perfect["perfect_grader"]
    assert perfect["share_of_best_possible"] == 1.0


def test_password_hash_roundtrip():
    h = auth.hash_password("hunter2")
    assert auth.verify_password("hunter2", h)
    assert not auth.verify_password("wrong", h)


def test_token_roundtrip_and_expiry():
    t = auth.make_token("packer", ttl=60)
    assert auth.check_token(t) == "packer"
    expired = auth.make_token("packer", ttl=-1)
    assert auth.check_token(expired) is None
    assert auth.check_token("not.a.token") is None


def test_login():
    assert auth.login("packer", "freshgrade") is not None
    assert auth.login("packer", "nope") is None
