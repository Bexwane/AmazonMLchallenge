import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.metric import f05, macro_f05  # noqa: E402


def test_official_example():
    # From the problem statement: P = 2/3, R = 1 -> 0.714
    assert abs(f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) - 0.7142857) < 1e-6


def test_singletons():
    assert f05(set(), set()) == 1.0
    assert f05({"S2-1"}, set()) == 0.0


def test_missed_and_wrong():
    assert f05(set(), {"S2-1"}) == 0.0
    assert f05({"S2-9"}, {"S2-1"}) == 0.0


def test_perfect_and_precision_weight():
    assert f05({"a", "b"}, {"a", "b"}) == 1.0
    # precision-heavy: half recall at full precision beats full recall at half precision
    assert f05({"a"}, {"a", "b"}) > f05({"a", "x"}, {"a"})


def test_macro_missing_prediction_is_empty():
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    assert macro_f05({}, truth) == 0.5
