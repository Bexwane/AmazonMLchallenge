"""Normalize a split once and cache it ("bake then eat")."""
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .io import read_split
from .normalize import normalize_address, normalize_name


def _normalize_chunk(args):
    names, addrs = args
    return ([normalize_name(x) for x in names], [normalize_address(x) for x in addrs])


def _normalize(df: pd.DataFrame, procs: int, chunk=100_000) -> pd.DataFrame:
    jobs = [(df["business_name"].iloc[i:i + chunk].tolist(), df["business_address"].iloc[i:i + chunk].tolist())
            for i in range(0, len(df), chunk)]
    if procs > 1 and len(jobs) > 1:
        with Pool(procs) as pool:
            res = pool.map(_normalize_chunk, jobs)
    else:
        res = [_normalize_chunk(j) for j in jobs]
    names = pd.DataFrame.from_records([r for n, _ in res for r in n], index=df.index)
    addrs = pd.DataFrame.from_records([r for _, a in res for r in a], index=df.index)
    return pd.concat([df, names, addrs], axis=1)


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """Arrow-backed strings: ~4x less memory than Python objects for the 10M-row S2/S3 table."""
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string[pyarrow]")
    return df


def load_prepared(data_dir, split: str, work_dir, procs=None):
    """Return normalized (s1, s23), reading the parquet cache when present."""
    cache = Path(work_dir) / "prepared" / split
    p1, p23 = cache / "s1.parquet", cache / "s23.parquet"
    if p1.exists() and p23.exists():
        return _compact(pd.read_parquet(p1)), _compact(pd.read_parquet(p23))
    procs = procs or os.cpu_count() or 1
    s1, s23 = read_split(data_dir, split)
    s1, s23 = _normalize(s1, procs), _normalize(s23, procs)
    cache.mkdir(parents=True, exist_ok=True)
    s1.to_parquet(p1, index=False)
    s23.to_parquet(p23, index=False)
    return _compact(s1), _compact(s23)
