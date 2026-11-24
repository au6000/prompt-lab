"""Rows: the rows file split by time, and the Design of one split.

A Design holds what the yardsticks need about the measured units, built once and aligned by
position: which rows, in which order, with which weights, cross-validation groups and halves.
The order is part of every recorded number (shuffles and zero-point draws index units by
position), so it is fixed here and nowhere else.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from .layout import Layout, PlabError, read_jsonl, sha256_file
from .task import SPLITS, TaskConfig, parse_time
from .yardsticks import Labels, rnd


@dataclass(frozen=True)
class Row:
    id: str
    t: datetime
    z: int
    value: float | None
    state: Mapping[str, Any]


@dataclass(frozen=True)
class SplitRows:
    """The rows of one split in time order; rows at the same time keep the order of the rows file."""
    split: str
    rows: tuple[Row, ...]
    sha256: str


@dataclass(frozen=True)
class Design:
    split: str
    rows: tuple[Row, ...]
    labels: Labels
    data_sha256: str

    def take(self, idx: np.ndarray) -> Design:
        lab = self.labels
        return replace(self, rows=tuple(self.rows[i] for i in idx),
                       labels=replace(lab, z=lab.z[idx], w=lab.w[idx], groups=lab.groups[idx], half=lab.half[idx],
                                      value=None if lab.value is None else lab.value[idx]))


def split_rows(task: TaskConfig, layout: Layout, *, force: bool) -> dict[str, Any]:
    """The rows file -> .plab/data/{train,eval,test}.jsonl by time. Rows outside every range are
    dropped and counted. The summary describes the training split only: evaluation and test
    outcomes stay out of sight."""
    if layout.split_file("train").exists() and not force:
        raise PlabError("already split (.plab/data/). Use --force to redo it before freezing")
    f = layout.root / task.rows.file
    if not f.exists():
        raise PlabError(f"rows file not found: {f}")
    R = task.rows
    out: dict[str, list] = {name: [] for name in SPLITS}
    seen: set[str] = set()
    dropped = 0
    with open(f, encoding="utf-8") as fh:
        for ln, text in enumerate(fh, 1):
            if not text.strip():
                continue
            where = f"{f.name}: line {ln}"
            try:
                r = json.loads(text)
            except json.JSONDecodeError:
                raise PlabError(f"{where} is not valid JSON")
            if not isinstance(r, dict):
                raise PlabError(f"{where} is not an object")
            missing = [k for k in (R.id, R.time, R.outcome, R.state, *([R.value] if R.value else [])) if k not in r]
            if missing:
                raise PlabError(f"{where} has no {', '.join(missing)} (see task.json rows)")
            rid, t, label, state, value = str(r[R.id]), r[R.time], r[R.outcome], r[R.state], r.get(R.value)
            if rid in seen:
                raise PlabError(f"{f.name}: duplicate id {rid!r}")
            seen.add(rid)
            try:
                task.outcome.index(label)
                dt = parse_time(t)
            except PlabError as e:
                raise PlabError(f"{where}: {e}")
            if R.value and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise PlabError(f"{where}: value must be a finite number")
            if not isinstance(state, dict):
                raise PlabError(f"{where}: state must be an object")
            line = {"id": rid, "t": t, "label": label, **({"value": float(value)} if R.value else {}), "state": state}
            for name, (a, b) in task.splits.items():
                if a <= dt < b:
                    out[name].append((dt, line))
                    break
            else:
                dropped += 1
    layout.split_file("train").parent.mkdir(parents=True, exist_ok=True)
    for name, items in out.items():
        items.sort(key=lambda x: x[0])
        tmp = layout.split_file(name).with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for _, line in items:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        tmp.replace(layout.split_file(name))
    train = [line for _, line in out["train"]]
    counts = np.bincount([task.outcome.index(line["label"]) for line in train], minlength=task.outcome.K)
    return {"rows": {name: len(items) for name, items in out.items()}, "dropped_outside_ranges": dropped,
            "train": {"class_counts": dict(zip(task.outcome.classes, counts.tolist())),
                      "all_in_value_per_1000": rnd(1000 * sum(x["value"] for x in train) / len(train), 1) if R.value and train else None,
                      "first": train[0]["t"] if train else None, "last": train[-1]["t"] if train else None}}


def load_split(task: TaskConfig, layout: Layout, split: str) -> SplitRows:
    f = layout.split_file(split)
    if not f.exists():
        raise PlabError("the data is not split yet: run `plab split` first")
    rows = tuple(Row(r["id"], parse_time(r["t"]), task.outcome.index(r["label"]), r.get("value"), r["state"]) for r in read_jsonl(f))
    if not rows:
        raise PlabError(f"the {split} split is empty")
    return SplitRows(split, rows, sha256_file(f))


def state_fields(task: TaskConfig, train: SplitRows) -> tuple[str, ...]:
    """The fields a prompt may give. For "all": the keys of the first 200 training rows in the
    order first seen, which is also the order of the JSON object a decider reads."""
    if task.state_fields != "all":
        return task.state_fields
    return tuple(dict.fromkeys(k for r in train.rows[:200] for k in r.state))


def _group(task: TaskConfig, r: Row) -> str:
    g = task.cv_groups
    if g == "month":
        return r.t.strftime("%Y-%m")
    if g == "week":
        year, week, _ = r.t.isocalendar()
        return f"{year}-W{week:02d}"
    if g == "day":
        return r.t.strftime("%Y-%m-%d")
    if g == "row":
        return r.id
    v = r.state.get(g)
    return "" if v is None else str(v)


def design(task: TaskConfig, split: SplitRows, *, sampled: bool = True) -> Design:
    """The units of one split. Evaluation and test: every row in time order, weight 1.

    Training with train_sample (and sampled): every row outside the reference class in time order,
    then a seeded pick of reference rows in time order, each carrying the weight of the reference
    rows it stands for. With two classes that is every positive row, then the picked negatives.
    """
    rows, w = list(split.rows), np.ones(len(split.rows))
    other = [r for r in rows if r.z != 0]
    ref = [r for r in rows if r.z == 0]
    ts = task.train_sample
    if split.split == "train" and ts and sampled and other and ref:
        k = min(len(ref), max(1, int(round(float(ts.reference_per_other) * len(other)))))
        pick = sorted(np.random.default_rng(ts.seed).choice(len(ref), k, replace=False))
        rows = other + [ref[p] for p in pick]
        w = np.array([1.0] * len(other) + [len(ref) / k] * k)
    groups = np.array([_group(task, r) for r in rows])
    if len(np.unique(groups)) < 2:
        raise PlabError(f"cv_groups {task.cv_groups!r} puts every {split.split} row in one group; cross-validation needs at least two "
                        "(use month, week, day, row or a state field)")
    halved = split.split == "train" and task.half_boundary is not None
    half = np.array([1 if r.t < task.half_boundary else 2 for r in rows]) if halved else np.zeros(len(rows), int)
    value = np.array([r.value for r in rows], float) if task.rows.value else None
    return Design(split.split, tuple(rows), Labels(np.array([r.z for r in rows]), task.outcome.K, w, groups, half, value),
                  split.sha256)


def breakeven(rows: tuple[Row, ...]) -> dict[str, float | None] | None:
    """For a two-class outcome with values: the positive rate at which taking every row is worth
    zero, from the mean value of each class. None unless positives lose and negatives win."""
    pos = [r.value for r in rows if r.z == 1]
    neg = [r.value for r in rows if r.z == 0]
    if not pos or not neg:
        return None
    mp, mn = sum(pos) / len(pos), sum(neg) / len(neg)
    if not mn > 0 > mp:
        return None
    return {"mean_value_pos": rnd(mp, 3), "mean_value_neg": rnd(mn, 3), "breakeven_pos_pct": rnd(100 * mn / (mn - mp), 2),
            "breakeven_neg_pct": rnd(100 * (-mp) / (mn - mp), 2)}


def period_key(t: datetime, by: str) -> str:
    if by == "year":
        return f"{t.year}"
    if by == "half":
        return f"{t.year}H{1 if t.month <= 6 else 2}"
    return f"{t.year}Q{(t.month - 1) // 3 + 1}"
