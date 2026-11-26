"""One definition of coverage, with a width-aware companion (Amendment 17, report-only).

For a coordinate (log tau1 or log p) of a stored coordinate-profile result, the 95% profile interval is read from the
profile (`classification.ci_low`, `ci_high`). An interval that is open on one side runs to the edge of the box, and a flat
profile spans the box. Every interval is therefore a finite interval inside the box, and four quantities are defined on
all subjects of a replica, with the truth known:

* `bounded`            the profile rises above delta on both sides inside the box (the paper's definition);
* `covers`             the true value lies inside the interval (this is the only coverage the paper calls coverage);
* `narrow_covers`      the interval is bounded and narrower than `NARROW_LOG_WIDTH` in log units and contains the truth;
* `score`              the interval score of Gneiting and Raftery for a 95% interval on the log scale: the width plus
                       (2/alpha) times the distance by which the truth lies outside the interval (lower is better).

An interval that runs to the box edge covers the truth almost for free, so `covers` alone flatters unbounded intervals.
`covers_given_narrow` reports coverage among the intervals that are bounded and narrow. The box log range is reported so
that the width of an uninformative interval can be read against it.
"""
from __future__ import annotations

import math

import numpy as np

ALPHA = 0.05
NARROW_LOG_WIDTH = 0.5          # log units: an interval narrower than a factor of exp(0.5), about 1.65


def interval(row: dict, key: str) -> tuple[float, float] | None:
    """The interval in natural units clipped to the box, or None when the stored profile has no truth or box."""
    entry = row["profiles"][key]
    box = row.get("box", {}).get(key)
    if box is None:
        return None
    lo_box, hi_box = math.exp(box[0]), math.exp(box[1])
    cls = entry["classification"]
    lo = cls.get("ci_low")
    hi = cls.get("ci_high")
    lo = lo_box if lo is None else max(float(lo), lo_box)
    hi = hi_box if hi is None else min(float(hi), hi_box)
    return lo, hi


def metrics(rows: list[dict], key: str) -> dict:
    n = covers = bounded = narrow = narrow_cov = 0
    scores, widths = [], []
    box_range = None
    for row in rows:
        entry = row["profiles"][key]
        truth = entry.get("truth")
        iv = interval(row, key)
        if truth is None or iv is None:
            continue
        lo, hi = iv
        box = row["box"][key]
        box_range = box[1] - box[0]
        n += 1
        width = math.log(hi / lo)
        c = lo <= truth <= hi
        covers += int(c)
        is_bounded = entry["classification"]["verdict"] == "identifiable"
        bounded += int(is_bounded)
        if is_bounded and width <= NARROW_LOG_WIDTH:
            narrow += 1
            narrow_cov += int(c)
        miss = max(math.log(lo / truth), 0.0) + max(math.log(truth / hi), 0.0)
        scores.append(width + (2.0 / ALPHA) * miss)
        widths.append(width)
    if n == 0:
        return {"n": 0}
    return {"n": n, "bounded": bounded, "covers": covers, "narrow": narrow, "narrow_covers": narrow_cov,
            "covers_given_narrow": (narrow_cov / narrow) if narrow else None,
            "median_score": float(np.median(scores)), "median_log_width": float(np.median(widths)),
            "box_log_range": box_range}
