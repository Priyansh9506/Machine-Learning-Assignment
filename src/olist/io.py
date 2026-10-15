"""Selective, typed CSV loaders for the Olist tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C


def load(table: str, usecols: list[str] | None | str = "default",
         data_dir: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load one source table with only the needed columns, compact dtypes and
    explicitly formatted timestamps.

    usecols="default" uses config.USECOLS; None loads every column.
    """
    if table not in C.FILES:
        raise KeyError(f"unknown table {table!r}; choose from {sorted(C.FILES)}")
    path = (data_dir or C.DATA_DIR) / C.FILES[table]
    cols = C.USECOLS[table] if usecols == "default" else usecols
    dtypes = {k: v for k, v in C.DTYPES[table].items() if cols is None or k in cols}
    df = pd.read_csv(path, usecols=cols, dtype=dtypes, nrows=nrows)
    if table in C.DATE_COLS:
        date_cols, fmt = C.DATE_COLS[table]
        for col in date_cols:
            if col in df.columns:
                df[col] = parse_dates(df[col], fmt)
    return df.rename(columns=C.RENAMES.get(table, {}))


def parse_dates(s: pd.Series, fmt: str) -> pd.Series:
    """Parse with an explicit format and fail loudly on any malformed value
    (missing values stay NaT)."""
    out = pd.to_datetime(s, format=fmt, errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        raise ValueError(f"{bad.sum()} values in {s.name!r} do not match {fmt!r}, "
                         f"e.g. {s[bad].iloc[0]!r}")
    return out


def file_checksums(data_dir: Path | None = None) -> dict[str, str]:
    """SHA-256 of every source CSV, so a rerun can prove it used the same data."""
    import hashlib
    out = {}
    for name in C.FILES.values():
        h = hashlib.sha256()
        with ((data_dir or C.DATA_DIR) / name).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[name] = h.hexdigest()
    return out


def file_inventory(data_dir: Path | None = None) -> pd.DataFrame:
    """File name, size and row count for every CSV without loading it into pandas."""
    rows = []
    for table, name in C.FILES.items():
        path = (data_dir or C.DATA_DIR) / name
        header = pd.read_csv(path, nrows=0).columns.tolist()
        # Physical lines overcount reviews (multi-line comments); parse that one.
        n = (len(pd.read_csv(path, usecols=[0])) if table == "reviews"
             else sum(1 for _ in path.open("rb")) - 1)
        rows.append({"table": table, "file": name, "size_mb": round(path.stat().st_size / 1e6, 2),
                     "rows": n, "n_cols": len(header)})
    return pd.DataFrame(rows).set_index("table")
