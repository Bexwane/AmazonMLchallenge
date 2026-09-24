"""Build a miniature train universe that preserves the full-data ratios.

Keep a random fraction f of Source-1 entities with all their matched S2/S3 records, plus
the same fraction of unmatched (distractor) S2/S3 records. Records owned by dropped S1
entities are dropped too, so the matched/distractor mix per S1 stays identical.

Also reports whether matched pairs ever cross countries (blocking-by-country check).

usage: python scripts/make_subset.py 0.01 [out_dir]
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.config import DATA_DIR, SEED  # noqa: E402

frac = float(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else f"data/subsets/f{frac:g}") / "train"
out.mkdir(parents=True, exist_ok=True)
src = DATA_DIR / "train"
rng = random.Random(SEED)

keep_s1, keep_ids = set(), {2: [], 3: []}
owned = {2: [], 3: []}
with open(src / "train_ground_truth.tsv", encoding="utf-8") as f, \
        open(out / "train_ground_truth.tsv", "w", encoding="utf-8", newline="\n") as g:
    g.write(next(f))
    for line in f:
        s1, _, rest = line.rstrip("\n").partition("\t")
        ids = rest.split(",") if rest else []
        take = rng.random() < frac
        if take:
            keep_s1.add(s1)
            g.write(line)
        for x in ids:
            s = int(x[1])
            owned[s].append(int(x[3:]))
            if take:
                keep_ids[s].append(int(x[3:]))
owned = {s: np.unique(np.asarray(v, np.int64)) for s, v in owned.items()}
keep_ids = {s: np.unique(np.asarray(v, np.int64)) for s, v in keep_ids.items()}

s1_country = {}
with open(src / "train_source1.tsv", encoding="utf-8") as f, \
        open(out / "train_source1.tsv", "w", encoding="utf-8", newline="\n") as g:
    g.write(next(f))
    for line in f:
        eid = line[:line.index("\t")]
        if eid in keep_s1:
            g.write(line)
            s1_country[eid] = line.rstrip("\n").rsplit("\t", 1)[1]

rec_country = {}
for s in (2, 3):
    n_keep = 0
    with open(src / f"train_source{s}.tsv", encoding="utf-8") as f, \
            open(out / f"train_source{s}.tsv", "w", encoding="utf-8", newline="\n") as g:
        g.write(next(f))
        for line in f:
            eid = line[:line.index("\t")]
            num = int(eid[3:])
            i = np.searchsorted(keep_ids[s], num)
            if i < len(keep_ids[s]) and keep_ids[s][i] == num:
                g.write(line); n_keep += 1
                rec_country[eid] = line.rstrip("\n").rsplit("\t", 1)[1]
                continue
            j = np.searchsorted(owned[s], num)
            is_owned = j < len(owned[s]) and owned[s][j] == num
            if not is_owned and rng.random() < frac:
                g.write(line); n_keep += 1
    print(f"S{s}: kept {n_keep}")

cross = total = 0
with open(out / "train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, _, rest = line.rstrip("\n").partition("\t")
        for x in (rest.split(",") if rest else []):
            total += 1
            cross += rec_country[x] != s1_country[s1]
print(f"S1 kept {len(keep_s1)}; matched pairs {total}; cross-country pairs {cross}")
