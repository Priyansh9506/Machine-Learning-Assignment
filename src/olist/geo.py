"""Compact zip-prefix -> coordinate mapping, built once from the 1M-row geolocation table.

The raw geolocation table is never joined to anything. Instead:
1. drop exact duplicate rows,
2. drop points outside Brazil's bounding box,
3. take the median lat/lng per 5-digit zip prefix (robust to stray points),
4. cache the ~19k-row result as parquet.
Unmatched zips fall back to the median of their 3-digit prefix, then to the state median.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .io import load

ZIP_MAP_PATH = C.INTERIM_DIR / "zip_centroids.parquet"


def build_zip_map(geo: pd.DataFrame | None = None, cache: Path | None = ZIP_MAP_PATH,
                  refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return (zip_map indexed by 5-digit prefix, cleaning stats)."""
    if cache is not None and cache.exists() and not refresh and geo is None:
        stats_path = cache.with_suffix(".stats.json")
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        return pd.read_parquet(cache), stats
    geo = load("geolocation") if geo is None else geo
    stats = {"raw_rows": len(geo)}
    g = geo.drop_duplicates()
    stats["exact_duplicates_dropped"] = stats["raw_rows"] - len(g)
    in_box = (g["geolocation_lat"].between(*C.BRAZIL_BBOX["lat"])
              & g["geolocation_lng"].between(*C.BRAZIL_BBOX["lng"]))
    stats["outside_brazil_dropped"] = int((~in_box).sum())
    g = g[in_box]
    zm = (g.groupby("geolocation_zip_code_prefix", observed=True)
            .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"),
                 n_points=("geolocation_lat", "size"),
                 state=("geolocation_state", lambda s: s.mode().iat[0]))
            .rename_axis("zip_prefix"))
    zm["state"] = zm["state"].astype(str)
    stats["zip_prefixes"] = len(zm)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        zm.to_parquet(cache)
        cache.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    return zm, stats


def lookup_coords(zips: pd.Series, states: pd.Series, zip_map: pd.DataFrame) -> pd.DataFrame:
    """Coordinates for each (zip, state) with a hierarchical fallback.

    Returns lat, lng and geo_source in {"zip5", "zip3", "state", "missing"}, aligned
    to the input index.
    """
    zips = zips.astype(str)
    zip3 = zip_map.groupby(zip_map.index.str[:3])[["lat", "lng"]].median()
    st = zip_map.groupby("state")[["lat", "lng"]].median()

    out = pd.DataFrame(index=zips.index, columns=["lat", "lng"], dtype="float64")
    src = pd.Series("missing", index=zips.index, dtype=object)
    levels = [("state", states.astype(str), st), ("zip3", zips.str[:3], zip3), ("zip5", zips, zip_map)]
    # Fill coarse -> fine so the finest available level wins.
    for name, keys, table in levels:
        hit = keys.isin(table.index)
        out.loc[hit, ["lat", "lng"]] = table.loc[keys[hit], ["lat", "lng"]].to_numpy()
        src[hit] = name
    out["geo_source"] = src
    return out


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
