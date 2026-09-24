"""TSV readers/writers. Files have no quoting, so parse with QUOTE_NONE and keep empty strings."""
import csv
from pathlib import Path

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
                     na_filter=False, encoding="utf-8")
    assert list(df.columns) == COLS, df.columns
    return df


def read_split(data_dir, split: str):
    """Return (s1, s23) frames. s23 stacks Source 2 and Source 3 with a `src` column (2/3)."""
    d = Path(data_dir) / split
    s1 = read_source(d / f"{split}_source1.tsv")
    s2 = read_source(d / f"{split}_source2.tsv")
    s3 = read_source(d / f"{split}_source3.tsv")
    s2["src"] = 2
    s3["src"] = 3
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23["src"] = s23["src"].astype("int8")
    return s1, s23


def read_ground_truth(path) -> dict:
    """{s1_id: set(matched ids)}; singletons map to an empty set."""
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            gt[s1] = set(rest.split(",")) if rest else set()
    return gt


def write_id_lists(path, s1_ids, lists, header2: str):
    """Write one row per S1 id; `lists` maps s1_id -> iterable of S2/S3 ids (missing = empty)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header2}\n")
        for s1 in s1_ids:
            ids = lists.get(s1, ())
            # dict.fromkeys keeps order and removes duplicates
            f.write(f"{s1}\t{','.join(dict.fromkeys(ids))}\n")
