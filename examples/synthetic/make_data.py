"""Make a synthetic data set for the prompt-lab walkthrough (no real data, no key needed).

    python examples/synthetic/make_data.py [--out examples/synthetic/rows.jsonl] [--n 12000] [--seed 7]

Each row is a timestamped "alert" between 2018-01-01 and 2025-12-31 (weekdays). Taking the alert is
worth +2 when it works (label 0) and -50 when it fails (label 1), so the rows you take must fail
less than 2/52 = 3.85% of the time to break even. About 5% fail. Three things are planted:

(a) a narrow, structural effect that never changes: alerts between 13:00 and 15:59 UTC fail
    1.6 times as often as the rest;
(b) a trend effect that flips in 2022: before 2022 a strong move toward the alert direction
    (move_24h_toward > 0.6 x vol_24h) makes failure 2.4 times as likely; from 2022 it halves the
    failure rate, and a strong move the other way makes failure 2.4 times as likely;
(c) volatility (vol_24h, band_width and the moves) grows 2.5 times during the first half of 2022.

With the splits in task.json (train 2018-2021, eval 2022-2023, test 2024-2025) the trend rule looks
good in both training halves and fails on the evaluation split, while the time window survives.
The other fields are noise.
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

T0 = datetime(2018, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
FLIP = datetime(2022, 1, 1, tzinfo=timezone.utc)
RAMP_END = datetime(2022, 7, 1, tzinfo=timezone.utc)
DAYS = "Mon Tue Wed Thu Fri Sat Sun".split()


def vol_scale(t):
    if t < FLIP:
        return 1.0
    if t >= RAMP_END:
        return 2.5
    return 1.0 + 1.5 * (t - FLIP).total_seconds() / (RAMP_END - FLIP).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("rows.jsonl")))
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    days = [T0 + timedelta(days=k) for k in range((T1 - T0).days)]
    days = [d for d in days if d.weekday() < 5]
    rows = []
    for _ in range(a.n):
        d = days[rng.integers(len(days))]
        u = rng.random()
        if u < 0.66:
            minute = rng.uniform(0, 1440)
        elif u < 0.88:
            minute = rng.uniform(7 * 60, 17 * 60)
        else:
            minute = rng.uniform(13 * 60, 16 * 60)
        t = d + timedelta(minutes=int(minute))
        rows.append(t)
    rows.sort()
    out = []
    pos = 0
    for k, t in enumerate(rows):
        sc = vol_scale(t)
        vol = round(float(sc * np.exp(rng.normal(np.log(20), 0.35))), 1)
        band = round(float(vol * rng.uniform(0.35, 0.65)), 1)
        z = float(rng.normal(0, 0.5))
        move24 = round(z * vol, 1)
        move5 = round(float(abs(rng.normal(0, 1)) * band * 0.15), 1)
        hhmm = f"{t.hour:02d}:{t.minute:02d}"
        window = "13:00" <= hhmm <= "15:59"
        p = 0.034
        if window:
            p *= 1.6
        zz = move24 / vol if vol else 0.0
        if t < FLIP:
            if zz > 0.6:
                p *= 2.4
        else:
            if zz > 0.6:
                p *= 0.5
            elif zz < -0.6:
                p *= 2.4
        label = int(rng.random() < p)
        pos += label
        nxt = int(rng.integers(0, 4000)) if rng.random() < 0.9 else None
        state = {
            "event": "price alert (synthetic): price touched the outer band",
            "side": "BUY" if rng.random() < 0.5 else "SELL",
            "time_utc": hhmm,
            "weekday": DAYS[t.weekday()],
            "vol_24h": vol,
            "band_width": band,
            "move_24h_toward": move24,
            "move_5min_toward": move5,
            "alerts_last_3h": int(rng.poisson(0.6)),
            "minutes_to_next_release": nxt,
            "is_month_end": t.day >= 28 and (t + timedelta(days=4)).month != t.month,
            "spread": round(float(rng.choice([0.2, 0.3, 0.4, 0.6], p=[0.5, 0.3, 0.15, 0.05])), 1),
            "rsi_14": round(float(rng.uniform(15, 85)), 1),
            "dist_round_number": round(float(rng.uniform(0, 50)), 1),
            "last_12_moves": [round(float(x), 1) for x in rng.normal(0, 0.1 * vol, 12)],
        }
        out.append({"id": f"a{k:05d}", "t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "state": state, "label": label,
                    "value": -50.0 if label else 2.0})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} rows ({pos} positive, {100 * pos / len(out):.2f}%) to {a.out}")


if __name__ == "__main__":
    main()
