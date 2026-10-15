"""Score new, unseen orders with the deployed late-delivery model.

The raw CSVs are cleaned, joined and turned into features with exactly the same code as
training (olist.dataset), including the as-of seller/route history, which only uses
outcomes known before each order's purchase time. Each order then gets a calibrated late
probability, its own break-even threshold from the cost model, and an action.

Examples
    python predict.py                              # every order still in transit (outcome unknown)
    python predict.py --since 2018-08-01           # every order purchased on or after a date
    python predict.py --order-ids my_ids.txt       # specific orders, one id per line
    python predict.py --data-dir path/to/csvs --out predictions.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from olist import config as C  # noqa: E402
from olist.dataset import build_order_dataset, load_sources  # noqa: E402
from olist.geo import build_zip_map  # noqa: E402
from olist.io import load  # noqa: E402
from olist.scoring import load_deployed, score  # noqa: E402

FINAL_STATUSES = {"delivered", "canceled", "unavailable"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR, help="folder with the 9 Olist CSVs")
    ap.add_argument("--since", help="score orders purchased on or after this date (YYYY-MM-DD)")
    ap.add_argument("--order-ids", type=Path, help="text file with one order_id per line")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "predictions.csv")
    ap.add_argument("--top", type=int, default=15, help="rows to print")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    t0 = time.perf_counter()
    model, meta = load_deployed()
    custom_dir = args.data_dir.resolve() != C.DATA_DIR.resolve()
    src = load_sources(args.data_dir)
    zip_map = build_zip_map(load("geolocation", data_dir=args.data_dir), cache=None)[0] if custom_dir else None
    df, _ = build_order_dataset(src, zip_map)

    if args.order_ids:
        ids = {line.strip() for line in args.order_ids.read_text().splitlines() if line.strip()}
        sel, what = df[df.order_id.isin(ids)], f"{len(ids)} requested order ids"
    elif args.since:
        sel, what = df[df.order_purchase_timestamp >= pd.Timestamp(args.since)], f"orders purchased since {args.since}"
    else:
        sel, what = df[~df.order_status.isin(FINAL_STATUSES)], "orders still in transit (not delivered, cancelled or unavailable)"
    if sel.empty:
        print(f"no orders matched: {what}")
        return 1

    out = score(sel, model, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_exp = int((out.action == "EXPEDITE").sum())
    print(f"model: {meta['model']} ({meta['trained_on']})")
    print(f"scored {len(out):,} {what} in {time.perf_counter() - t0:.0f}s")
    print(f"recommend EXPEDITE for {n_exp:,} orders ({n_exp / len(out):.1%}); mean late probability {out.p_late.mean():.3f}")
    print(f"saved -> {args.out}\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(out.head(args.top)[["order_id", "route", "p_late", "break_even_p", "action", "top_risk_factors"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
