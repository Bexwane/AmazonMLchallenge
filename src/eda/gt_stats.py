"""Streaming ground-truth statistics (low memory: ids held as int64 numpy arrays)."""
import sys, collections
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else "D:/6ab10eb3b23ba_student_resource/student_resource/dataset"

def main():
    n2_hist, n3_hist, tot_hist = collections.Counter(), collections.Counter(), collections.Counter()
    s2_ids, s3_ids = [], []
    n = 0
    with open(f"{DATA}/train/train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            ids = rest.split(",") if rest else []
            a = [int(x[3:]) for x in ids if x.startswith("S2-")]
            b = [int(x[3:]) for x in ids if x.startswith("S3-")]
            s2_ids.extend(a); s3_ids.extend(b)
            n2_hist[min(len(a), 10)] += 1; n3_hist[min(len(b), 10)] += 1; tot_hist[min(len(ids), 15)] += 1
            n += 1
    print("S1 rows in GT:", n)
    print("total-matches hist (capped 15):", sorted(tot_hist.items()))
    print("S2-matches hist (capped 10):", sorted(n2_hist.items()))
    print("S3-matches hist (capped 10):", sorted(n3_hist.items()))
    for name, arr in (("S2", s2_ids), ("S3", s3_ids)):
        arr = np.asarray(arr, dtype=np.int64)
        u, c = np.unique(arr, return_counts=True)
        print(f"{name}: matched ids={len(arr)} unique={len(u)} ids-in->1-S1-list={int((c>1).sum())}")

main()
