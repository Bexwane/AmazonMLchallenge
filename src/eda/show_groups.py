"""Print sample GT groups with the actual records (streaming, low memory)."""
import sys, random
DATA = "D:/6ab10eb3b23ba_student_resource/student_resource/dataset"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 25
random.seed(0)
groups = []
with open(f"{DATA}/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i > 400000: break
        if random.random() < K / 400000:
            s1, _, rest = line.rstrip("\n").partition("\t")
            groups.append((s1, rest.split(",") if rest else []))
want = {g[0] for g in groups} | {x for g in groups for x in g[1]}
rec = {}
for src in (1, 2, 3):
    with open(f"{DATA}/train/train_source{src}.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            eid = line.partition("\t")[0]
            if eid in want:
                rec[eid] = line.rstrip("\n").split("\t")[1:]
for s1, ids in groups:
    print("=" * 100)
    print(s1, "|", " | ".join(rec.get(s1, ["?"])))
    for x in ids:
        print("   ", x, "|", " | ".join(rec.get(x, ["?"])))
