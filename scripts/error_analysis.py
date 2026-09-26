"""Hypothesis tests on an out-of-fold error sample (oof_errors.parquet from `ber.run stack`).

  python scripts/error_analysis.py --data DATASET_DIR --errors oof_errors.parquet --out reports/errors.md

s1_idx / r_idx index the raw train files (Source 2 rows, then Source 3 rows), so the full ground truth
tells us, for every false positive, who the S2/S3 record really belongs to, and for every false negative,
what the S1's other true records look like. Tested:
  H1 word-level competition: an FP record belongs to another S1 whose name differs from ours by a word
  H2 entity grouping: an FN record shares its exact address with one of the S1's other true records;
     an FP record shares its exact address with records of another entity (or unmatched ones)
  H3 house numbers: FP pairs whose first house numbers differ although both have one
"""
import argparse
import csv
import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.buckets import _md  # noqa: E402

ABBR = {"street": "st", "road": "rd", "avenue": "ave", "drive": "dr", "lane": "ln", "boulevard": "blvd", "court": "ct",
        "place": "pl", "highway": "hwy", "north": "n", "south": "s", "east": "e", "west": "w", "suite": "ste",
        "rue": "r", "floor": "fl", "number": "no"}
LEGAL = {"llc", "inc", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private", "llp", "pc", "sarl",
         "sas", "eurl", "sa", "sci", "the", "and", "of", "l", "c"}


def rd(p):
    return pd.read_csv(p, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def norm_addr(s: pd.Series) -> pd.Series:
    s = s.str.lower().str.replace(r"[^0-9a-z\u0080-￿]+", " ", regex=True)
    return s.map(lambda x: " ".join(sorted(ABBR.get(t, t) for t in x.split())))


def name_words(x: str) -> set:
    return {t for t in re.sub(r"[^0-9a-z\u0080-￿]+", " ", x.lower()).split() if t not in LEGAL}


def first_num(x: str) -> str:
    m = re.search(r"\d+", x)
    return m.group(0) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--errors", required=True)
    ap.add_argument("--out", default="reports/error_analysis.md")
    a = ap.parse_args()
    d = Path(a.data) / "train"
    s1 = rd(d / "train_source1.tsv")
    s23 = pd.concat([rd(d / "train_source2.tsv"), rd(d / "train_source3.tsv")], ignore_index=True)
    gt = rd(d / "train_ground_truth.tsv")
    g = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    g = g[g["m"] != ""]
    pos1 = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    pos23 = pd.Series(np.arange(len(s23)), index=s23["entity_id"])
    owner = np.full(len(s23), -1, np.int64)
    owner[pos23[g["m"]].to_numpy()] = pos1[g["source1_entity_id"]].to_numpy()
    s23["addr_n"] = norm_addr(s23["business_address"])
    s1["addr_n"] = norm_addr(s1["business_address"])
    addr_groups = s23.groupby("addr_n").indices
    true_of = pd.Series(pos23[g["m"]].to_numpy(), index=pos1[g["source1_entity_id"]].to_numpy()).groupby(level=0).agg(list)

    E = pd.read_parquet(a.errors)
    E["owner"] = owner[E["r_idx"].to_numpy()]
    L = []

    def P(*x):
        L.append(" ".join(str(v) for v in x))

    P("# Error analysis\n")
    P(E.groupby(["kind", "country"]).size().unstack(fill_value=0).reset_index().pipe(_md), "\n")
    for k in ("FN", "FP"):
        P(f"## {k} modes\n")
        P(E[E.kind == k].groupby(["mode", "country"]).size().unstack(fill_value=0).assign(
            total=lambda t: t.sum(1)).sort_values("total", ascending=False).reset_index().pipe(_md), "\n")

    # ---- FP: who does the record really belong to? ----
    F = E[E.kind == "FP"].copy()
    F["belongs"] = np.where(F["owner"] < 0, "unmatched (no S1)", "another S1")
    P("## FP: true owner of the wrongly merged record\n")
    P(F.groupby(["belongs", "country"]).size().unstack(fill_value=0).reset_index().pipe(_md), "\n")
    o = F[F["owner"] >= 0]
    if len(o):
        on = s1["business_name"].to_numpy()[o["owner"].to_numpy()]
        oa = s1["addr_n"].to_numpy()[o["owner"].to_numpy()]
        ours_a = s1["addr_n"].to_numpy()[o["s1_idx"].to_numpy()]
        rw = [name_words(x) for x in o["r_name"]]
        ow, uw = [name_words(x) for x in on], [name_words(x) for x in o["s1_name"]]
        h1 = [bool((r & (b - u))) for r, b, u in zip(rw, ow, uw)]  # r has a word of the owner that we lack
        P(f"- H1: record contains a word of its true owner's name that our S1 lacks: **{np.mean(h1):.1%}** of {len(o)}")
        P(f"- owner S1 has exactly our S1's address (same-address different entity): **{np.mean(oa == ours_a):.1%}**")
        P(f"- owner name words == our name words (indistinguishable names): {np.mean([a_ == b_ for a_, b_ in zip(ow, uw)]):.1%}")
    ln, rn = F["s1_addr"].map(first_num), F["r_addr"].map(first_num)
    both = (ln != "") & (rn != "")
    P(f"- H3: both have a house number and the first numbers differ: **{np.mean(both & (ln != rn)):.1%}** of FP "
      f"(both-have-number {both.mean():.1%})")
    same = F["r_idx"].map(lambda r: len(addr_groups.get(s23["addr_n"].iat[r], ())))
    P(f"- FP record's exact address shared by >=2 S2/S3 records: {np.mean(same >= 2):.1%}")
    ra = s23["addr_n"].to_numpy()[F["r_idx"].to_numpy()]
    la = s1["addr_n"].to_numpy()[F["s1_idx"].to_numpy()]
    P(f"- FP record's address == our S1 address (normalized): {np.mean(ra == la):.1%}\n")

    # ---- FN: do the S1's other true records vouch for the missed one? ----
    N = E[E.kind == "FN"].copy()
    sib_addr, sib_num = [], []
    for s, r in zip(N["s1_idx"], N["r_idx"]):
        sibs = [x for x in true_of.get(s, []) if x != r]
        sa = {s23["addr_n"].iat[x] for x in sibs}
        sib_addr.append(s23["addr_n"].iat[r] in sa and s23["addr_n"].iat[r] != "")
        sib_num.append(first_num(s23["business_address"].iat[r]) in {first_num(s23["business_address"].iat[x]) for x in sibs}
                       and first_num(s23["business_address"].iat[r]) != "")
    P("## FN: evidence from the S1's other true records\n")
    P(f"- H2: missed record has exactly the address of a sibling true record: **{np.mean(sib_addr):.1%}** of {len(N)}")
    P(f"- missed record shares its first house number with a sibling: {np.mean(sib_num):.1%}")
    rA = s23["addr_n"].to_numpy()[N["r_idx"].to_numpy()]
    lA = s1["addr_n"].to_numpy()[N["s1_idx"].to_numpy()]
    P(f"- missed record's address == our S1 address (normalized): {np.mean(rA == lA):.1%}")
    P(f"- p1 of misses: median {N['p1'].median():.3f}, share p1 > 0.3: {(N['p1'] > 0.3).mean():.1%}\n")
    for k, T in (("FP", F), ("FN", N)):
        P(f"## {k} examples\n")
        cols = ["country", "mode", "p_final", "s1_name", "s1_addr", "r_name", "r_addr"]
        X = T.sample(min(40, len(T)), random_state=0)[cols].copy()
        if k == "FP":
            X["owner_name"] = [s1["business_name"].iat[o_] if o_ >= 0 else "-" for o_ in T.loc[X.index, "owner"]]
            X["owner_addr"] = [s1["business_address"].iat[o_] if o_ >= 0 else "-" for o_ in T.loc[X.index, "owner"]]
        P(X.pipe(_md), "\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
