"""Do unmatched S2/S3 records share an address with any S1? (house no + street word search)"""
import random, re
DATA = "D:/6ab10eb3b23ba_student_resource/student_resource/dataset"
random.seed(7)
matched = set()
with open(f"{DATA}/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        rest = line.rstrip("\n").partition("\t")[2]
        if rest: matched.update(rest.split(","))
probes = []
for s in (2, 3):
    with open(f"{DATA}/train/train_source{s}.tsv", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i > 200000: break
            p = line.rstrip("\n").split("\t")
            if p[0] not in matched and p[3] == "US" and random.random() < 0.0006:
                m = re.search(r"\b(\d{2,6})\s+([A-Za-z]{4,})", p[2])
                if m: probes.append((p, m.group(1), m.group(2).lower()))
probes = probes[:14]
del matched
hits = [[] for _ in probes]
with open(f"{DATA}/train/train_source1.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        low = line.lower()
        for i, (p, num, st) in enumerate(probes):
            if num in low and st in low and re.search(rf"\b{num}\s+{st}", low):
                hits[i].append(line.rstrip("\n"))
for (p, num, st), h in zip(probes, hits):
    print("=" * 100); print("DISTRACTOR", " | ".join(p))
    for x in h[:6]: print("     S1 same addr:", x.replace("\t", " | "))
