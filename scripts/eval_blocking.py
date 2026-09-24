"""Measure blocking recall ceiling on a (sub)set of the training data.

usage: python scripts/eval_blocking.py <data_dir> [k_comb k_name df_cap]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.blocking import generate_candidates  # noqa: E402
from ber.io import read_ground_truth  # noqa: E402
from ber.labels import blocking_report, label_pairs  # noqa: E402
from ber.prepare import load_prepared  # noqa: E402

data_dir = Path(sys.argv[1])
k_comb, k_name, df_cap = (int(x) for x in (sys.argv[2:5] if len(sys.argv) >= 5 else (40, 10, 2000)))
work = data_dir.parent / "work"
t = time.time()
s1, s23 = load_prepared(data_dir, "train", work)
print(f"prepared {len(s1)} S1, {len(s23)} S2/S3 in {time.time() - t:.0f}s")
truth = read_ground_truth(data_dir / "train" / "train_ground_truth.tsv")
t = time.time()
cand = generate_candidates(s1, s23, k_comb=k_comb, k_name=k_name, df_cap=df_cap)
print(f"blocking {time.time() - t:.0f}s")
y = label_pairs(cand, s1, s23, truth)
print(json.dumps(blocking_report(cand, y, s1, s23, truth), indent=1))
