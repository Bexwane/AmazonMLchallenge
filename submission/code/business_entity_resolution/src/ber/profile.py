"""E011: record profiles for twin-business detection.

Error analysis on train (E009 OOF sample joined to the ground truth): most false merges are synthetic twin
businesses that belong to no Source-1 entity. A twin copies a real business and changes its LEGAL FORM
(53% of twin pairs add a legal form the S1 lacks vs 14% of true pairs; e.g. Private Limited -> Public Limited,
LLC -> Co, none -> LLC), swaps or adds a business word (Products / Investment, + Industries), and nudges the
house number's last digits (2921 -> 2942). The pair features so far could not see any of it: name_core drops
legal forms and token_set_ratio scores "X" vs "X LLC" or "X" vs "X Industries" as 100.

Per record (computed once per split, vectorized pair features afterwards):
  legal : bitmask of canonical legal forms read from name_clean tokens (typo and transliteration tolerant)
  wsig  : 64-bit signature of the name's business-word skeletons (legal forms and generator noise words out)
  noise : bitmask of generator noise words (Center, Services, Shri, Sri, Mr, www, ...)
  hnum  : first house number as an integer (-1 when none), hlen its digit count
"""
import zlib

import numpy as np
import pandas as pd

from .normalize import skeleton

LEGAL_CODES = {
    "PVT": "private pvt pvte prvt prv privat praivet praivate prayvet pvtltd",
    "LTD": "limited ltd limitad limted lmtd pvtltd",
    "PUB": "public",
    "LLP": "llp", "LLC": "llc", "PLLC": "pllc", "INC": "inc incorporated", "CORP": "corp corporation",
    "CO": "co company cie compagnie", "LP": "lp", "PC": "pc", "PLC": "plc",
    "SARL": "sarl", "SAS": "sas", "SASU": "sasu", "EURL": "eurl", "SA": "sa", "SCI": "sci", "SNC": "snc",
    "EI": "ei", "ETS": "ets etablissements etablissement", "SOC": "societe ste", "GMBH": "gmbh",
}
BIT = {c: 1 << i for i, c in enumerate(LEGAL_CODES)}
TOK2LEGAL = {t: BIT[c] for c, ws in LEGAL_CODES.items() for t in ws.split()}
TOK2LEGAL["pvtltd"] = BIT["PVT"] | BIT["LTD"]
LEGAL_SKELS = {skeleton("private"): BIT["PVT"], skeleton("limited"): BIT["LTD"]}  # "Priavate", "Limtied"
NOISE = ("center centre services service partners com www shri sri smt mr mrs dr ms the id dba and of m s "
         "district metro").split()
NOISE_BIT = {w: 1 << i for i, w in enumerate(NOISE)}
_POP = np.bitwise_count if hasattr(np, "bitwise_count") else None


def _popcount(x):
    if _POP is not None:
        return _POP(x).astype(np.float32)
    x = x.astype(np.uint64)
    c = np.zeros(len(x), np.float32)
    while x.any():
        c += (x & np.uint64(1)).astype(np.float32)
        x >>= np.uint64(1)
    return c


def _record(clean: str):
    legal = noise = 0
    sig = 0
    for t in clean.split():
        b = TOK2LEGAL.get(t)
        if b is not None:
            legal |= b
            continue
        nb = NOISE_BIT.get(t)
        if nb is not None:
            noise |= nb
            continue
        k = skeleton(t)
        lb = LEGAL_SKELS.get(k)
        if lb is not None and len(t) >= 5:
            legal |= lb
            continue
        if k:
            sig |= 1 << (zlib.crc32(k.encode()) & 63)
    return legal, noise, sig


def _hnum(nums: str):
    t = nums.split(" ", 1)[0] if nums else ""
    d = ""
    for ch in t:
        if ch.isdigit():
            d += ch
        else:
            break
    return (int(d[:15]), len(d)) if d else (-1, 0)


def record_profiles(df: pd.DataFrame) -> dict:
    rec = [_record(x) for x in df["name_clean"].astype(str).tolist()]
    hn = [_hnum(x) for x in df["addr_nums"].astype(str).tolist()]
    return {"legal": np.fromiter((r[0] for r in rec), np.uint32, len(rec)),
            "noise": np.fromiter((r[1] for r in rec), np.uint32, len(rec)),
            "wsig": np.fromiter((r[2] for r in rec), np.uint64, len(rec)),
            "hnum": np.fromiter((h[0] for h in hn), np.int64, len(hn)),
            "hlen": np.fromiter((h[1] for h in hn), np.int8, len(hn))}


_MEMO = {}


def profile_arrays(s1: pd.DataFrame, s23: pd.DataFrame) -> dict:
    """Profiles of both sides plus name+legal-form ambiguity counts (memoized per frame pair)."""
    key = (id(s1), id(s23))
    if key in _MEMO:
        return _MEMO[key]
    L, R = record_profiles(s1), record_profiles(s23)
    # how many S1 entities carry this exact (core name, legal forms): rivals a legal-form-aware matcher sees
    k1 = s1["name_core"].astype(str).to_numpy(object) + "|" + L["legal"].astype(str).astype(object)
    k23 = s23["name_core"].astype(str).to_numpy(object) + "|" + R["legal"].astype(str).astype(object)
    codes, _ = pd.factorize(np.concatenate([k1, k23]))
    c1, c23 = codes[:len(k1)], codes[len(k1):]
    cnt = np.bincount(c1, minlength=codes.max() + 1).astype(np.float32)
    L["namelg_s1dup"], R["namelg_s1cnt"] = cnt[c1], cnt[c23]
    from .features import name_amb_arrays
    _MEMO.clear()
    _MEMO[key] = {"L": L, "R": R, "amb": name_amb_arrays(s1, s23)}
    return _MEMO[key]


def profile_features(s1_idx, r_idx, PA, amb=True) -> pd.DataFrame:
    """Pair features from record profiles. amb=False leaves out the core-name ambiguity counts (stage 1 v5+
    already has them under the same names)."""
    L, R = PA["L"], PA["R"]
    lm, rm = L["legal"][s1_idx], R["legal"][r_idx]
    ln, rn = L["noise"][s1_idx], R["noise"][r_idx]
    lw, rw = L["wsig"][s1_idx], R["wsig"][r_idx]
    a, b = L["hnum"][s1_idx], R["hnum"][r_idx]
    la, lb = L["hlen"][s1_idx].astype(np.float32), R["hlen"][r_idx].astype(np.float32)
    both = (a >= 0) & (b >= 0)
    diff = np.abs(a - b).astype(np.float64)
    mag = np.where(both & (diff > 0), np.floor(np.log10(np.maximum(diff, 1))) + 1, 0).astype(np.float32)
    pub = np.uint32(BIT["PUB"])
    pl = np.uint32(BIT["PVT"] | BIT["LTD"])
    F = pd.DataFrame({
        "lg_l_n": _popcount(lm), "lg_r_n": _popcount(rm),
        "lg_add": _popcount(rm & ~lm), "lg_drop": _popcount(lm & ~rm),
        "lg_add_other": _popcount(rm & ~lm & ~pl),  # added form other than Private/Limited (noise-prone)
        "lg_same": (lm == rm).astype(np.float32),
        "lg_pub_flip": (((lm & pub) > 0) != ((rm & pub) > 0)).astype(np.float32),
        "w_extra": _popcount(rw & ~lw), "w_miss": _popcount(lw & ~rw), "w_same": (lw == rw).astype(np.float32),
        "nz_extra": _popcount(rn & ~ln),
        "hn_both": both.astype(np.float32), "hn_eq": (both & (a == b)).astype(np.float32),
        "hn_diff_mag": mag, "hn_diff_lead": np.where(both, mag - la, 0).astype(np.float32),
        "hn_len_eq": (la == lb).astype(np.float32),
        "namelg_s1dup": L["namelg_s1dup"][s1_idx], "namelg_r_s1cnt": R["namelg_s1cnt"][r_idx],
        "l_name_s1dup": PA["amb"]["l_name_s1dup"][s1_idx], "r_name_s1cnt": PA["amb"]["r_name_s1cnt"][r_idx],
        "r_name_s23cnt": PA["amb"]["r_name_s23cnt"][r_idx],
    })
    return F if amb else F.drop(columns=["l_name_s1dup", "r_name_s1cnt", "r_name_s23cnt"])


def profile_codes(PA) -> dict:
    """S2/S3 group codes for stage-2 consensus: legal-form set, and the full profile (house number, legal
    forms, business words). A twin's records share their own profile; a typo record is alone in its group."""
    R = PA["R"]
    if "codes" not in R:
        prof, _ = pd.factorize(pd.MultiIndex.from_arrays([R["hnum"], R["legal"], R["wsig"]]))
        R["codes"] = {"legal": R["legal"].astype(np.int64), "prof": prof.astype(np.int64)}
    return R["codes"]
