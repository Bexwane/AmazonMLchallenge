"""Error buckets for blocking: measurable properties of true pairs the blocker missed vs. found.

Every feature is computed from the data only (no labels beyond "is this a true pair"). Misses are
compared with a random sample of found pairs, so each bucket gets a lift = share among misses /
share among hits. Failure modes are assigned by ordered rules (first match wins) so they partition
the misses; the multi-label flag table shows overlaps.
"""
import re

import numpy as np
import pandas as pd

_SCRIPTS = [(0x0900, "Devanagari"), (0x0980, "Bengali"), (0x0A00, "Gurmukhi"), (0x0A80, "Gujarati"),
            (0x0B00, "Oriya"), (0x0B80, "Tamil"), (0x0C00, "Telugu"), (0x0C80, "Kannada"), (0x0D00, "Malayalam")]
_INDIC = re.compile(r"[ऀ-ൿ]")
_POST = re.compile(r"\b\d{5,6}\b")


def script_of(name: str) -> str:
    m = _INDIC.search(name)
    if not m:
        return "Latin" if re.search(r"[A-Za-z]", name) else "other"
    cp = ord(m.group(0))
    return next(n for start, n in reversed(_SCRIPTS) if cp >= start)


def _jac(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if A and B else 0.0


def _last_comp(addr: str) -> str:
    parts = [p.strip().lower() for p in addr.split(",") if p.strip()]
    return parts[-1] if parts else ""


def pair_table(pairs: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame, name_freq: pd.Series) -> pd.DataFrame:
    """Bucket features for true pairs (`pairs` has s1_idx, r_idx and diagnosis columns)."""
    L = s1.iloc[pairs["s1_idx"].to_numpy()].reset_index(drop=True)
    R = s23.iloc[pairs["r_idx"].to_numpy()].reset_index(drop=True)
    t = pairs.reset_index(drop=True).copy()
    t["country"] = L["country"].to_numpy()
    t["source"] = "S" + R["src"].astype(str).to_numpy()
    rn, ra = R["business_name"].tolist(), R["business_address"].tolist()
    la = L["business_address"].tolist()
    t["r_script"] = [script_of(x) for x in rn]
    t["r_native"] = R["name_native"].to_numpy(bool)
    t["r_addr_empty"] = R["addr_empty"].to_numpy(bool)
    t["r_alias_or_web"] = R["name_has_alias"].to_numpy(bool) | R["name_is_web"].to_numpy(bool)
    lc, rc = L["name_core"].tolist(), R["name_core"].tolist()
    t["s1_name_tokens"] = [len(x.split()) for x in lc]
    t["r_name_tokens"] = [len(x.split()) for x in rc]
    t["s1_name_chars"] = [len(x) for x in lc]
    t["name_tok_jac"] = [_jac(a, b) for a, b in zip(lc, rc)]
    t["name_skel_jac"] = [_jac(a, b) for a, b in zip(L["name_skel"].tolist(), R["name_skel"].tolist())]
    t["name_compact_eq"] = L["name_compact"].to_numpy() == R["name_compact"].to_numpy()
    lA, rA = L["addr_clean"].tolist(), R["addr_clean"].tolist()
    t["s1_addr_tokens"] = [len(x.split()) for x in lA]
    t["r_addr_tokens"] = [len(x.split()) for x in rA]
    t["addr_tok_jac"] = [_jac(a, b) for a, b in zip(lA, rA)]
    t["num_jac"] = [_jac(a, b) for a, b in zip(L["addr_nums"].tolist(), R["addr_nums"].tolist())]
    t["s1_has_num"] = [bool(x) for x in L["addr_nums"].tolist()]
    t["r_has_num"] = [bool(x) for x in R["addr_nums"].tolist()]
    lp = [set(_POST.findall(x)) for x in la]
    rp = [set(_POST.findall(x)) for x in ra]
    t["postcode_both"] = [bool(a) and bool(b) for a, b in zip(lp, rp)]
    t["postcode_eq"] = [bool(a & b) for a, b in zip(lp, rp)]
    t["last_comp_eq"] = [_last_comp(a) == _last_comp(b) for a, b in zip(la, ra)]
    key = L["country"].astype(str).to_numpy() + "|" + L["name_core"].astype(str).to_numpy()
    t["s1_name_freq"] = pd.Series(key).map(name_freq).fillna(0).to_numpy(np.int64)
    t["s1_name"], t["s1_addr"] = L["business_name"].to_numpy(), la
    t["r_name"], t["r_addr"] = rn, ra
    return t


def name_frequency(s23: pd.DataFrame) -> pd.Series:
    """How many pool records share a normalized name, keyed by 'country|name_core'."""
    key = s23["country"].astype(str) + "|" + s23["name_core"].astype(str)
    return key.value_counts()


# ordered rules: the first matching rule names the miss
MODES = [
    ("empty_address_record", lambda t: t["r_addr_empty"]),
    ("native_script_name", lambda t: t["r_native"]),
    ("alias_or_web_name_form", lambda t: t["r_alias_or_web"]),
    ("name_replaced_same_address", lambda t: (t["name_tok_jac"] == 0) & (t["name_skel_jac"] == 0) & (t["addr_tok_jac"] >= 0.5)),
    ("name_replaced_address_changed", lambda t: (t["name_tok_jac"] == 0) & (t["name_skel_jac"] == 0)),
    ("address_rewritten_name_kept", lambda t: (t["addr_tok_jac"] < 0.25) & (t["name_tok_jac"] >= 0.5)),
    ("common_name_crowd", lambda t: t["s1_name_freq"] >= 20),
    ("no_key_survives_df_cap", lambda t: t["comb"] <= 0),
    ("partial_name_and_address_noise", lambda t: (t["name_tok_jac"] < 0.5) & (t["addr_tok_jac"] < 0.5)),
    ("similar_but_outranked", lambda t: t["comb"] > 0),
]


def assign_modes(t: pd.DataFrame) -> pd.Series:
    mode = pd.Series("other", index=t.index, dtype=object)
    free = np.ones(len(t), bool)
    for name, rule in MODES:
        m = np.asarray(rule(t), bool) & free
        mode[m] = name
        free &= ~m
    return mode


FLAGS = {
    "r_native": lambda t: t["r_native"], "r_addr_empty": lambda t: t["r_addr_empty"],
    "r_alias_or_web": lambda t: t["r_alias_or_web"], "source_S2": lambda t: t["source"] == "S2",
    "name_tok_jac=0": lambda t: t["name_tok_jac"] == 0, "name_tok_jac<0.5": lambda t: t["name_tok_jac"] < 0.5,
    "name_skel_jac=0": lambda t: t["name_skel_jac"] == 0, "name_compact_eq": lambda t: t["name_compact_eq"],
    "addr_tok_jac<0.25": lambda t: t["addr_tok_jac"] < 0.25, "addr_tok_jac>=0.75": lambda t: t["addr_tok_jac"] >= 0.75,
    "num_jac=0 (both have numbers)": lambda t: t["s1_has_num"] & t["r_has_num"] & (t["num_jac"] == 0),
    "r_has_no_number": lambda t: ~t["r_has_num"], "postcode_both": lambda t: t["postcode_both"],
    "postcode_eq": lambda t: t["postcode_eq"], "last_component_eq (state/region)": lambda t: t["last_comp_eq"],
    "s1_name_tokens=1": lambda t: t["s1_name_tokens"] == 1, "s1_name_tokens>=4": lambda t: t["s1_name_tokens"] >= 4,
    "s1_name_chars<=6": lambda t: t["s1_name_chars"] <= 6, "r_addr_tokens<=3": lambda t: t["r_addr_tokens"] <= 3,
    "common_name (freq>=20)": lambda t: t["s1_name_freq"] >= 20, "rare_name (freq<=1)": lambda t: t["s1_name_freq"] <= 1,
    "no_surviving_key (comb=0)": lambda t: t["comb"] <= 0,
}


def flag_table(miss: pd.DataFrame, hit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, f in FLAGS.items():
        a, b = float(np.mean(f(miss))), float(np.mean(f(hit)))
        rows.append({"flag": name, "share_of_misses": a, "share_of_hits": b, "lift": a / b if b > 0 else np.inf})
    return pd.DataFrame(rows).sort_values("share_of_misses", ascending=False)


def _md(df: pd.DataFrame, floatfmt=3) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: f"{x:.{floatfmt}f}")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in r) + " |")
    return "\n".join(lines)


def write_report(path, summary: dict, miss: pd.DataFrame, hit: pd.DataFrame):
    miss = miss.copy()
    miss["mode"] = assign_modes(miss)
    n = len(miss)
    top = []
    for mode, g in miss.groupby("mode"):
        cd = g["country"].value_counts(normalize=True).round(3).to_dict()
        sd = g["source"].value_counts(normalize=True).round(3).to_dict()
        top.append({"failure_mode": mode, "missed_pairs": len(g), "share_of_misses": len(g) / n,
                    "country": cd, "source": sd})
    top = pd.DataFrame(top).sort_values("missed_pairs", ascending=False).head(10)
    out = ["# Error Buckets (full-density blocker)", "",
           f"Blocker: `{summary['blocker']}`. True pairs: {summary['true_pairs']:,}. "
           f"Missed (absent from the candidate set): {summary['absent']:,} "
           f"({summary['absent'] / summary['true_pairs']:.2%}). Found pairs compared against a random sample of "
           f"{len(hit):,} found pairs.", "",
           "## Present vs absent (mandatory split)", "", _md(pd.DataFrame(summary["presence"])), "",
           "A reranker can only recover the *present* rows; *absent* rows need new retrieval.", "",
           "## Absent pairs: mechanism", "",
           f"* no key survives the df cap (name_cos + addr_cos = 0): {summary['absent_no_key']:.2%} of absent",
           f"* scored but outranked: {1 - summary['absent_no_key']:.2%} of absent",
           f"* full-pool rank of scored-but-absent pairs (sample of {summary['rank_sample']:,}): "
           f"{summary['absent_rank_quantiles']}", "",
           "## Top-10 measured failure modes (ordered rules, first match wins, partition of absent pairs)", "",
           _md(top), "",
           "## Flags: share among misses vs among found pairs", "",
           _md(flag_table(miss, hit)), "",
           "## By country / source / script", ""]
    for col in ("country", "source", "r_script"):
        mm = miss[col].value_counts()
        hh = hit[col].value_counts(normalize=True)
        tab = pd.DataFrame({"misses": mm, "share_of_misses": mm / n, "share_of_hits": hh}).fillna(0).reset_index()
        out += [_md(tab), ""]
    out += ["## Examples (5 per mode)", ""]
    cols = ["country", "source", "s1_name", "s1_addr", "r_name", "r_addr", "name_cos", "addr_cos"]
    for mode in top["failure_mode"]:
        out += [f"### {mode}", "", _md(miss.loc[miss["mode"] == mode, cols].head(5)), ""]
    open(path, "w", encoding="utf-8").write("\n".join(out))
    return top
