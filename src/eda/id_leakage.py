"""Leakage audit: are entity-id numbers or file row positions informative of matches?"""
import numpy as np
DATA = "D:/6ab10eb3b23ba_student_resource/student_resource/dataset"
def rowpos(path):
    ids, pos = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            ids.append(int(line[3:line.index("\t")])); pos.append(i)
    ids = np.asarray(ids, np.int64); order = np.argsort(ids)
    return ids[order], np.asarray(pos, np.int64)[order]
def lookup(tbl, q):
    ids, pos = tbl; j = np.searchsorted(ids, q); ok = (j < len(ids)) & (ids[np.minimum(j, len(ids)-1)] == q)
    return np.where(ok, pos[np.minimum(j, len(ids)-1)], -1)
s1tbl = rowpos(f"{DATA}/train/train_source1.tsv"); s2tbl = rowpos(f"{DATA}/train/train_source2.tsv")
a, b = [], []
with open(f"{DATA}/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 300000: break
        s1, _, rest = line.rstrip("\n").partition("\t")
        for x in rest.split(","):
            if x.startswith("S2-"): a.append(int(s1[3:])); b.append(int(x[3:]))
a = np.asarray(a); b = np.asarray(b)
print("pairs", len(a), "dup S1 id numbers?", len(s1tbl[0]) - len(np.unique(s1tbl[0])))
print("corr(S1 id, S2 id) =", np.corrcoef(a, b)[0, 1])
d = np.abs(a - b); rnd = np.abs(a - np.random.permutation(b))
print("median |id diff| true vs shuffled:", np.median(d), np.median(rnd))
pa, pb = lookup(s1tbl, a), lookup(s2tbl, b)
fa, fb = pa / len(s1tbl[0]), pb / len(s2tbl[0])
print("corr(row-frac S1, row-frac S2) =", np.corrcoef(fa, fb)[0, 1])
print("median |row-frac diff| true vs shuffled:", np.median(np.abs(fa - fb)), np.median(np.abs(fa - np.random.permutation(fb))))
