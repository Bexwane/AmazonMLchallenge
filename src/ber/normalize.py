"""Deterministic text normalization for business names and addresses.

Country-agnostic by design: abbreviation tables are merged across locales so an unseen
country (France in test) still gets sensible canonicalization.
"""
import re
import unicodedata

# 2: France-only rules (’, N°, departement -> region, French legal forms); train output unchanged
NORM_VERSION = 2

# ---------------------------------------------------------------------------
# Indic scripts -> rough Latin. All major Indic Unicode blocks share the ISCII
# layout, so one table indexed by (codepoint - block_start) covers Devanagari,
# Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam.
# ---------------------------------------------------------------------------
_INDIC_BLOCKS = range(0x0900, 0x0D80, 0x80)
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "dh", 0x5E: "f", 0x5F: "y",
}
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
           0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
           0x60: "ri", 0x61: "li"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
           0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li"}
_MODS = {0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_INDIC_RE = re.compile(r"[ऀ-ൿ]")


def _indic_offset(ch):
    cp = ord(ch)
    if 0x0900 <= cp < 0x0D80:
        return cp & 0x7F
    return None


def romanize_indic(text: str) -> str:
    """Transliterate Indic-script runs to Latin; non-Indic characters pass through unchanged."""
    if not _INDIC_RE.search(text):
        return text
    out = []
    pending_a = False  # inherent vowel after a consonant
    for ch in text:
        off = _indic_offset(ch)
        if off is None:
            if pending_a:
                pending_a = False  # schwa deletion at word end
            out.append(ch)
            continue
        if off in _CONS:
            if pending_a:
                out.append("a")
            out.append(_CONS[off])
            pending_a = True
        elif off in _MATRAS:
            out.append(_MATRAS[off])
            pending_a = False
        elif off == _VIRAMA:
            pending_a = False
        elif off == _NUKTA:
            continue
        elif off in _VOWELS:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_VOWELS[off])
        elif off in _MODS:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_MODS[off])
        elif 0x66 <= off <= 0x6F:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(str(off - 0x66))
        else:
            if pending_a:
                out.append("a")
                pending_a = False
    return "".join(out)


def fold(text: str) -> str:
    """Romanize Indic script, strip accents, lowercase. Typographic apostrophes become ASCII (France
    in test writes L’AERODROME where Source 1 has l'Aerodrome; train has no ’)."""
    text = romanize_indic(unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'"))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


# ---------------------------------------------------------------------------
# Phonetic consonant skeleton: makes romanized Indic and English spellings meet
# (e.g. "praaivet" / "private" -> "prvt", "kanstrakshans" / "constructions").
# ---------------------------------------------------------------------------
_SKEL_SUBS = [("tion", "shn"), ("ph", "f"), ("x", "ks"), ("q", "k"), ("c", "k"), ("w", "v"), ("v", "b"), ("z", "j")]  # b/v merge: Bengali has one letter for both
_H_AFTER_CONS = re.compile(r"(?<=[bcdfgjklmnpqrstvxz])h")
_VOWEL_RE = re.compile(r"[aeiou]")
_DUP_RE = re.compile(r"(.)\1+")


def skeleton(token: str) -> str:
    t = re.sub(r"[^a-z]", "", token)
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    t = _H_AFTER_CONS.sub("", t)
    t = _VOWEL_RE.sub("", t)
    return _DUP_RE.sub(r"\1", t)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
LEGAL = {
    "private", "pvt", "pvtltd", "limited", "ltd", "llp", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "plc", "pc", "pllc", "lp", "ltda", "gmbh", "sarl", "sas", "sasu",
    "sa", "eurl", "sci", "snc", "cie", "ms", "the", "and", "of", "et", "du", "de", "la", "le", "les",
    "des",
}
# French forms seen only in test (Ets <-> Etablissements, Cie <-> Compagnie, Ste <-> Societe, EI). Kept out
# of LEGAL_SKEL: their skeletons ("skt", "ts") would also delete Shakti / Scott / Tosh from name_skel.
LEGAL_FR = {"ets", "etablissements", "etablissement", "compagnie", "societe", "ei"}
# spelled-out dotted forms collapse before tokenizing: l.l.c. -> llc, s.a.r.l. -> sarl, m/s -> ms
_DOTTED = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?\.?)")
_ALIAS_RE = re.compile(r"\s*(?:\bformerly(?: known as)?:?|\bf/k/a\b|\bfka\b|\ba/k/a\b|\baka\b|\bt/a\b|"
                       r"\bd/b/a\b|\bdba\b|\btrading as\b|\bdoing business as\b)\s*")
_ID_NOISE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|#\s*\d+")
_WEB_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|in|net|org|co\.in|fr|co|biz|info)\b")
_NONWORD = re.compile(r"[^a-z0-9]+")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})

LEGAL_SKEL = {skeleton(w) for w in LEGAL | {"limited", "private", "llp"}} - {""}
LEGAL = LEGAL | LEGAL_FR


def _tokens(text: str):
    return [t for t in _NONWORD.split(text) if t]


def _deleet(tok: str) -> str:
    # "orth0pedic", "va1ue", "5atya": mixed letter/digit tokens are OCR-style typos
    if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        return tok.translate(_LEET)
    return tok


def normalize_name(raw: str) -> dict:
    s = fold(raw)
    s = s.replace("m/s", " ms ").replace("&", " and ")
    s = _DOTTED.sub(lambda m: m.group(1).replace(".", ""), s)
    s = _ID_NOISE.sub(" ", s)
    web = _WEB_RE.search(s)
    is_web = bool(web) or s.strip().startswith(("@", "#"))
    parts = _ALIAS_RE.split(s, maxsplit=1)
    has_alias = len(parts) > 1
    main = parts[-1] if has_alias else s  # the original name follows "formerly:"/"t/a"/...
    alias = parts[0] if has_alias else ""
    if web:
        main = main.replace(web.group(0), web.group(1))
    toks = [_deleet(t) for t in _tokens(main)]
    core = [t for t in toks if t not in LEGAL] or toks
    return {
        "name_clean": " ".join(toks),
        "name_core": " ".join(core),
        "name_compact": "".join(core),
        "name_skel": " ".join(k for k in (skeleton(t) for t in core) if k and k not in LEGAL_SKEL),
        "name_alias": " ".join(_tokens(alias)),
        "name_is_web": is_web,
        "name_has_alias": has_alias,
        "name_native": bool(_INDIC_RE.search(raw)),
    }


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
_ABBR = {}
for canon, variants in {
    "st": "street str saint sainte ste",  # the noise generator swaps St <-> Saint
    "rd": "road", "dr": "drive drv", "ave": "avenue av avn avenu", "ln": "lane", "ct": "court crt",
    "pl": "place plc", "blvd": "boulevard bd boul bvd", "cir": "circle", "trl": "trail",
    "pkwy": "parkway", "hwy": "highway", "ter": "terrace terr", "sq": "square", "n": "north",
    "s": "south", "e": "east", "w": "west", "apt": "apartment appt", "fl": "floor flr",
    "bldg": "building", "no": "number num nº n°", "near": "nr", "opp": "opposite", "rue": "r",
    "chemin": "ch chem", "allee": "all", "imp": "impasse", "quai": "q", "rte": "route",
    "cours": "crs", "sec": "sector", "col": "colony", "hno": "h.no", "po": "p.o",
}.items():
    for v in variants.split():
        _ABBR[v] = canon
_ORD_WORDS = {w: str(i) for i, w in enumerate(
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
_ADDR_DROP = {"null", "city", "pmb", "unit", "c", "o", "at", "and", "of", "the", "du", "de", "la",
              "le", "les", "des", "d", "l", "bis", "ter"}
_NUM_RE = re.compile(r"^0*(\d+)(?:st|nd|rd|th|e|er|eme)?$")

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}


# France (test only): Source 1 names the region, Source 2/3 often the departement instead.
FR_DEPARTMENTS = {
    "nord": "hauts-de-france", "pas-de-calais": "hauts-de-france", "pas de calais": "hauts-de-france",
    "gironde": "nouvelle-aquitaine", "loire-atlantique": "pays de la loire", "loire atlantique": "pays de la loire",
}
_NUMERO = re.compile(r"\bn\s*°")  # "N°24 R ..." (France): Source 1 writes just "24 Rue ..."


def normalize_address(raw: str) -> dict:
    s = fold(raw).replace("<null>", " ").replace("#", " ")
    if "°" in s:
        s = _NUMERO.sub(" ", s)
    comps = [c.strip() for c in s.split(",") if c.strip() and c.strip() != "null"]
    toks, nums = [], []
    for c in comps:
        c = US_STATES.get(c, c)  # full US state name component -> 2-letter code
        c = FR_DEPARTMENTS.get(c, c)
        for t in _tokens(c.replace("'", "")):
            t = _ORD_WORDS.get(t, t)
            m = _NUM_RE.match(t)
            if m:
                nums.append(m.group(1))
                toks.append(m.group(1))
                continue
            t = _ABBR.get(t, t)
            if t in _ADDR_DROP:
                continue
            toks.append(_deleet(t) if not t.isdigit() else t)
    alpha = [t for t in toks if not t.isdigit()]
    return {
        "addr_clean": " ".join(toks),
        "addr_nums": " ".join(nums),
        "addr_alpha": " ".join(alpha),
        "addr_skel": " ".join(k for k in (skeleton(t) for t in alpha) if k),
        "addr_empty": not comps,
        "addr_ncomp": len(comps),
    }
