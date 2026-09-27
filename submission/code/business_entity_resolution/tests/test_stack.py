import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.stack import p_context, sibling_features  # noqa: E402


def test_p_context_basic():
    s1 = np.array([0, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.6, 0.1], np.float32)
    P = p_context(s1, p)
    assert P["p1_rank_s1"].tolist() == [1, 3, 2, 1]
    assert np.allclose(P["p1_second_s1"], [0.6, 0.6, 0.6, 0.0])
    assert np.allclose(P["p1_margin"], [0.3, -0.7, -0.3, 0.1])
    assert np.allclose(P["p1_sum_s1"], [1.7, 1.7, 1.7, 0.1])


def test_siblings_compare_with_other_confident_candidates():
    s23 = pd.DataFrame({"addr_clean": ["12 main st", "12 main st", "99 elm rd"],
                        "name_core": ["acme", "akme", "zeta"], "name_skel": ["km", "km", "jt"],
                        "addr_alpha": ["main st", "main st", "elm rd"], "addr_nums": ["12", "12", "99"]})
    s1 = np.array([0, 0, 0])
    r = np.array([0, 1, 2])
    p = np.array([0.95, 0.1, 0.1], np.float32)
    S = sibling_features(s1, r, p, s23, top=3, min_p=0.2)
    assert S["sib_addr_max"].tolist()[1] == 1.0   # record 1 shares the address of the confident record 0
    assert S["sib_addr_max"].tolist()[2] < 0.5
    assert S["sib_hnum_max"].tolist() == [0.0, 1.0, 0.0]
    assert S["sib_n"].tolist() == [0.0, 1.0, 1.0]  # record 0 has no confident sibling other than itself
