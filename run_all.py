"""Rebuild the whole project from the raw CSVs in one command.

Checks the data checksums, executes notebooks 01 -> 09 in order (each one saves its own
figures, result tables and models), then runs the test suite.

    python run_all.py                  # everything
    python run_all.py --from 06        # resume from notebook 06
    python run_all.py --only 03 07     # just these notebooks
    python run_all.py --skip-tests
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
CHECKSUMS = ROOT / "reports" / "data_checksums.json"


def check_data() -> None:
    from olist.io import file_checksums
    try:
        now = file_checksums()
    except FileNotFoundError as exc:
        sys.exit(f"missing data file: {exc.filename}\nPut the 9 Olist CSVs in Dataset/ (see README).")
    if not CHECKSUMS.exists():
        print("no reference checksums found; skipping the data check")
        return
    ref = json.loads(CHECKSUMS.read_text())
    diff = [f for f in ref if now.get(f) != ref[f]]
    print("data checksums: identical to the reference run" if not diff else
          f"WARNING: {len(diff)} file(s) differ from the reference run (results will not match exactly): {diff}")


def run_notebook(path: Path) -> float:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    t = time.perf_counter()
    nb = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)
    return time.perf_counter() - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="01", help="first notebook number to run")
    ap.add_argument("--only", nargs="*", help="run only these notebook numbers")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    check_data()
    notebooks = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    chosen = [nb for nb in notebooks if (nb.name[:2] in args.only if args.only else nb.name[:2] >= args.start)]
    total = time.perf_counter()
    for nb in chosen:
        print(f"running {nb.name} ...", end=" ", flush=True)
        print(f"{run_notebook(nb):.0f}s")
    print(f"notebooks done in {time.perf_counter() - total:.0f}s")
    if not args.skip_tests:
        return subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
