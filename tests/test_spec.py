from fg import spec


def test_clean_fruit_is_class_I():
    assert spec.grade(0.0, 70.0, 0.0, 0.0) == spec.CLASS_I


def test_big_defect_is_reject():
    assert spec.grade(0.30, 70.0, 0.0, 0.0) == spec.REJECT


def test_worst_category_wins():
    # perfect except a heavy bruise -> reject, driven by defect
    g = spec.grade(0.20, 70.0, 0.0, 0.0)
    assert g == spec.REJECT
    assert spec.worst_category(0.20, 70.0, 0.0, 0.0) == "defect"


def test_size_band():
    assert spec.grade(0.0, 70.0, 0.0, 0.0) == spec.CLASS_I     # in band
    assert spec.grade(0.0, 57.0, 0.0, 0.0) == spec.CLASS_II    # small but sellable
    assert spec.grade(0.0, 45.0, 0.0, 0.0) == spec.REJECT      # far too small


def test_reasons_lists_the_fault():
    r = spec.reasons(0.20, 70.0, 0.0, 0.0)
    assert r and "defect" in r[0]


def test_grade_is_monotonic_in_defect():
    grades = [spec.grade(f, 70.0, 0.0, 0.0) for f in (0.0, 0.05, 0.2)]
    assert grades == sorted(grades)
