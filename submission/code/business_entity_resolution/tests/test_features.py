import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.features import _rank_gap, context_features  # noqa: E402


def test_rank_gap_matches_pandas():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 50, 5000)
    s = rng.integers(0, 8, 5000).astype(np.float32) / 8  # many ties
    rank, gap = _rank_gap(s, g)
    ref = pd.Series(-s).groupby(g).rank(method="min").to_numpy(np.float32)
    mx = pd.Series(s).groupby(g).transform("max").to_numpy(np.float32)
    assert np.array_equal(rank, ref)
    assert np.allclose(gap, s - mx)


def test_context_rows_subset_equals_full():
    rng = np.random.default_rng(1)
    n = 3000
    cand = pd.DataFrame({"s1_idx": rng.integers(0, 100, n), "r_idx": rng.integers(0, 300, n),
                         "name_cos": rng.random(n).astype(np.float32), "addr_cos": rng.random(n).astype(np.float32),
                         "rrf": rng.random(n).astype(np.float32)})
    rows = rng.random(n) < 0.3
    full = context_features(cand)
    sub = context_features(cand, rows)
    pd.testing.assert_frame_equal(full[rows], sub)
