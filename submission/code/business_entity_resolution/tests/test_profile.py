import numpy as np
import pandas as pd

from ber.normalize import normalize_address, normalize_name
from ber.profile import BIT, profile_arrays, profile_features


def _frame(rows):
    return pd.DataFrame([{**normalize_name(n), **normalize_address(a), "business_name": n} for n, a in rows])


def test_legal_forms_and_twin_signals():
    s1 = _frame([("Seven Systems Private Limited", "59 G.N. Chetty Road, Chennai"),
                 ("Border Metro Broadband", "12 Main Street, Austin, TX"),
                 ("Kairos Sport EURL", "106 Avenue du Pont Loby, Dunkerque")])
    s23 = _frame([("Seven Systems Public Limited", "59 G.N. Chetty Road, Chennai"),   # twin: Private -> Public
                  ("SEVEN SYSTEMS PVT LTD", "59 GN CHETTY RD, CHENNAI"),            # true, abbreviated
                  ("Border Metro Broadband LLC", "14 Main St, Austin, TX"),         # twin: + LLC, 12 -> 14
                  ("Kairos Sport SAS", "106 Av du Pont Loby, Dunkerque"),           # twin: EURL -> SAS
                  ("Seven Systems Priavate Limited", "59 Chetty Road, Chennai")])   # true, typo in Private
    PA = profile_arrays(s1, s23)
    assert PA["L"]["legal"][0] == BIT["PVT"] | BIT["LTD"]
    assert PA["R"]["legal"][4] == BIT["PVT"] | BIT["LTD"]
    F = profile_features(np.array([0, 0, 1, 2, 0]), np.array([0, 1, 2, 3, 4]), PA)
    assert F["lg_pub_flip"].tolist() == [1, 0, 0, 0, 0]
    assert F["lg_same"].tolist() == [0, 1, 0, 0, 1]
    assert F["lg_add"].tolist()[2:4] == [1, 1]
    assert F["hn_diff_mag"][2] == 1 and F["hn_eq"][1] == 1
    assert F["w_extra"].tolist()[:2] == [0, 0]
