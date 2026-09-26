import numpy as np
import pandas as pd

from ber.dense import _Fake, dense_context_arrays, pair_dense_cos, texts


def test_pair_cos_matches_direct_computation():
    s1 = pd.DataFrame({"business_name": ["Sharma Traders", "शर्मा ट्रेडर्स", "Cafe Lune"],
                       "business_address": ["12 MG Road, Pune", "", "3 rue Paul Bert, Lyon"]})
    s23 = pd.DataFrame({"business_name": ["Sharma Trading Co", "Café Lune", "Cafe Lune", "Other"],
                        "business_address": ["12 M.G. Rd, Pune", "3 Rue Paul-Bert, Lyon", "", "x"]})
    si = np.array([0, 2, 2, 1, 0, 2], np.int32)
    ri = np.array([0, 1, 2, 0, 3, 3], np.int32)
    enc = _Fake()
    D = pair_dense_cos(s1, s23, si, ri, enc, r_chunk=2, pair_chunk=1, log=lambda *a: None)
    for f in ("name", "full"):
        E1, E2 = enc.encode(texts(s1, np.arange(3), f), 0), enc.encode(texts(s23, np.arange(4), f), 0)
        want = (E1[si].astype(np.float32) * E2[ri].astype(np.float32)).sum(1)
        np.testing.assert_allclose(D[f].astype(np.float32), want, atol=2e-3)
    C = dense_context_arrays(si, ri, D)
    assert C["d_full_rank_s1"][0] in (1, 2) and set(C) >= {"d_name", "d_full", "d_full_gap_r", "d_name_rank_s1"}
