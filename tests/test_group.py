import numpy as np

from ber.stack import group_features


def test_twin_group_mass():
    # one S1 with a true group (addr code 0, three records) and a twin group (addr code 1, two records)
    s1 = np.zeros(5, np.int64)
    r = np.arange(5)
    p = np.array([0.9, 0.8, 0.3, 0.6, 0.5])
    codes = {"addr": np.array([0, 0, 0, 1, 1]), "hnum": np.array([-1, -1, 2, 3, 3])}
    G = group_features(s1, r, p, codes)
    np.testing.assert_allclose(G["grp_addr_n"], [3, 3, 3, 2, 2])
    np.testing.assert_allclose(G["grp_addr_psum_oth"], [1.1, 1.2, 1.7, 0.5, 0.6], atol=1e-6)
    np.testing.assert_allclose(G["grp_addr_gap"], [0, 0, 0, 0.9, 0.9], atol=1e-6)
    np.testing.assert_allclose(G["grp_hnum_n"], [1, 1, 1, 2, 2])  # empty codes are groups of their own
