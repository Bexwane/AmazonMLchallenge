"""Look at unmatched S2/S3 records (distractors) and S1 singletons: what look-alikes exist?"""
import random, re
DATA = "D:/6ab10eb3b23ba_student_resource/student_resource/dataset"
random.seed(1)
matched, singletons, owner = set(), [], {}
with open(f"{DATA}/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, _, rest = line.rstrip("\n").partition("\t")
        if rest:
            for x in rest.split(","): matched.add(x); owner[x] = s1
        elif random.random() < 12 / 123000:
            singletons.append(s1)
probes = []  # (probe_id, record)
def pick(path, k, cond):
    out = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i > 300000: break
            p = line.rstrip("\n").split("\t")
            if cond(p[0]) and random.random() < k / 80000: out.append(p)
    return out[:k]
probes += pick(f"{DATA}/train/train_source2.tsv", 8, lambda e: e not in matched)
probes += pick(f"{DATA}/train/train_source3.tsv", 6, lambda e: e not in matched)
ss = set(singletons)
with open(f"{DATA}/train/train_source1.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[0] in ss: probes.append(p)
del matched
def key(p):
    toks = [t for t in re.findall(r"[a-z0-9]+", p[1].lower()) if len(t) >= 4 and t not in {"private","limited","services","service","center","corporation","company","enterprises","india","trading"}]
    return max(toks, key=len) if toks else None
keys = [(key(p), p) for p in probes]
hits = {i: [] for i in range(len(keys))}
for s in (1, 2, 3):
    with open(f"{DATA}/train/train_source{s}.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            low = line.lower()
            for i, (k, p) in enumerate(keys):
                if k and k in low and len(hits[i]) < 12 and not line.startswith(p[0] + "\t"):
                    hits[i].append(line.rstrip("\n"))
for i, (k, p) in enumerate(keys):
    print("=" * 110); print("PROBE", " | ".join(p), " key=", k)
    for h in hits[i]:
        e = h.split("\t")[0]
        print("    ", h.replace("\t", " | "), "  <owner:", owner.get(e, "-"), ">")
