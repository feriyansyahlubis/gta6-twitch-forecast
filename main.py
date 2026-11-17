"""gta6-twitch-ccu-forecast-01: a launch-day Twitch viewer forecaster.

The program reads one case (schema "makefun/twitch-ccu-case/v1") and writes,
for each of hours 2..24 of the release day (UTC), five quantiles (5%, 25%, 50%,
75%, 95%) of that hour's concurrent Twitch viewers in the game's category.

Method (fitted on the 37 past big launches the kit ships as comparables, and
validated by leave-one-out over them):

  A launch's release day is a *shape* times a *scale*.  The scale is hour 1's
  print, which the case gives us.  The shape is the curve of log(viewers_h /
  viewers_1) over the day.  Past launches' shapes differ wildly -- a game
  that unlocked at midnight local time in New Zealand already has real
  viewers in hour 1 and ramps about 2x by the US evening; a game that
  unlocked at 15:00 UTC has a near-empty hour 1 and ramps 100x when it
  opens.  What predicts a launch's shape best is its own hour-1 scale:
  launches whose hour 1 drew a similar crowd ramp alike (leave-one-out
  pinball loss 0.254 on the big launches, against 0.321 for the house
  persistence baseline and 0.718 for the unweighted pooled quantiles).

  So, per hour h, the program takes the comparables' log-ratios
  log(viewers_h / viewers_1), weights each launch by a Gaussian kernel on
  log(hour-1 viewers) with bandwidth 1.0, and reads off the weighted median
  and the weighted interquartile range.  The forecast is log-normal around
  that median: q_tau = p1 * exp(median_h + z_tau * sigma_h), with
  sigma_h = IQR_h / 1.349 (floored at 0.08) and z the standard normal's
  quantiles.  The 5%-95% interval covers the truth about 90% of the time on
  the leave-one-out check, i.e. the spread is honest, not padded.

  Comparables that are not a launch day are left out: a curve needs some
  history before its day_start and a span of at most 450 hours (the kit's
  GTA VI pre-release curve, which starts at its premiere, has no history
  before it, and runs 26 days, fails both).  Every other comparable,
  including the practice launches the final case adds, is used.

  Nothing here reads the network, calls a model, or uses the clock: the same
  case always writes the same bytes.  Standard library only.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim: no network, no pip packages.  -P keeps this file's
directory OFF sys.path, so a module shipped beside main.py imports only
after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there): JSON {"schema":
    "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint",
    "levels", "known", "forecast", "history", "history_hourly", "streams",
    "categories_hourly", "comparables", "notes"}, where `known` holds hour
    1's print {hour: 1, from, until, viewers, channels, reads} and
    `forecast` lists the hours to forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most 1e9
    (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# How alike two launches' hour-1 crowds must be to count as alike: a
# Gaussian kernel on log(hour-1 viewers) with this bandwidth (in log units).
BANDWIDTH = 1.0
# The spread never goes below this, in log units per hour.
SIGMA_FLOOR = 0.08
# A comparable only counts as a launch day when it has this much history
# before its day_start and spans at most this many hours in all.
MIN_PRE_HISTORY = 20
MAX_SPAN_HOURS = 450
MIN_DAY_HOURS = 20


def parse_time(text):
    """An ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def weighted_quantile(pairs, p):
    """The weighted p-quantile of [(value, weight), ...]: the first value
    whose cumulative weight reaches p of the total.  Deterministic and
    identical to the check the model was validated with."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return 0.0
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= p * total:
            return value
    return ordered[-1][0]


def launch_day(comp):
    """A comparable's release-day curve as {hour: viewers} for hours 1..24,
    or None when it is not a launch day (too little history before its
    day_start, too long a span, or too few of the day's hours present)."""
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, ValueError):
        return None
    hourly = comp.get("hourly") or []
    if not hourly:
        return None
    times = []
    for point in hourly:
        try:
            times.append(parse_time(point["from"]))
        except (KeyError, ValueError):
            continue
    if not times:
        return None
    pre = sum(1 for t in times if t < day_start)
    span = (max(times) - min(times)).total_seconds() / 3600
    if pre < MIN_PRE_HISTORY or span > MAX_SPAN_HOURS:
        return None
    curve = {}
    for point in hourly:
        try:
            t = parse_time(point["from"])
            viewers = point["viewers"]
        except (KeyError, ValueError):
            continue
        hour = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and isinstance(viewers, (int, float)) and viewers > 0:
            curve[hour] = viewers
    if 1 not in curve or len(curve) < MIN_DAY_HOURS:
        return None
    return curve


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # The pool: every comparable that is a launch day, as (hour-1 viewers,
    # release-day curve).  The target's own scale decides who counts alike.
    pool = []
    for comp in case.get("comparables") or []:
        curve = launch_day(comp)
        if curve is not None:
            pool.append((max(curve[1], 1), curve))
    if not pool:
        raise ValueError(f"{name} has no usable comparables")

    log_p1 = math.log(p1)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        pairs = []
        for v1, curve in pool:
            if h in curve:
                weight = math.exp(-0.5 * ((log_p1 - math.log(v1)) / BANDWIDTH) ** 2)
                pairs.append((math.log(curve[h] / v1), weight))
        if not pairs:
            # No comparable reports this hour: fall back to the persistence
            # baseline's spread around hour 1's print.
            median, sigma = 0.0, 0.15 * math.sqrt(h - 1)
        else:
            median = weighted_quantile(pairs, 0.5)
            sigma = max((weighted_quantile(pairs, 0.75) - weighted_quantile(pairs, 0.25)) / 1.349, SIGMA_FLOOR)
        quantiles = [round2(p1 * math.exp(median + z * sigma)) for z in Z]
        # Rounding can only tie neighbours, never invert them; clamp to the
        # output's bounds all the same.
        quantiles = [min(max(q, 0.01), 1e9) for q in quantiles]
        quantiles.sort()
        hours.append({"hour": h, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
