"""Cardinality-checked merging. Every join in the project goes through safe_merge."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class JoinError(AssertionError):
    """Raised when a merge would violate its declared cardinality or multiply rows."""


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on=None, *, left_on=None,
               right_on=None, how: str = "left", validate: str = "m:1",
               name: str = "", allow_unmatched: bool = True) -> pd.DataFrame:
    """pd.merge with enforced cardinality, row-count guards and an unmatched-key log.

    - validate is required and must be one of "1:1" or "m:1": a left join of that
      kind can never multiply left rows. One-to-many relations must be aggregated
      to the left table's grain before calling this.
    - For how="left", the result must have exactly len(left) rows.
    - Unmatched left keys are counted and logged; allow_unmatched=False raises instead.
    """
    if validate not in {"1:1", "m:1", "one_to_one", "many_to_one"}:
        raise JoinError(f"[{name}] validate={validate!r}: aggregate one-to-many "
                        "relations before joining")
    lk = left_on if left_on is not None else on
    rk = right_on if right_on is not None else on
    try:
        out = left.merge(right, on=on, left_on=left_on, right_on=right_on, how=how,
                         validate=validate, indicator="_merge_status")
    except pd.errors.MergeError as exc:
        raise JoinError(f"[{name}] cardinality check failed: {exc}") from exc

    if how == "left" and len(out) != len(left):
        raise JoinError(f"[{name}] row count changed {len(left)} -> {len(out)}")
    unmatched = int((out["_merge_status"] == "left_only").sum())
    msg = (f"[{name or 'merge'}] {validate} {how} on {lk}->{rk}: rows {len(left)} -> "
           f"{len(out)}, unmatched left rows {unmatched}")
    log.info(msg)
    if unmatched and not allow_unmatched:
        raise JoinError(msg)
    out.attrs["join_report"] = {"name": name, "rows_before": len(left), "rows_after": len(out),
                                "unmatched_left": unmatched, "validate": validate}
    return out.drop(columns="_merge_status")
