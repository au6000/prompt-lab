"""The zero point: the same yardsticks, measured on random answers of the same shape as a prompt's.

The shape is a Signature: the kind and number of options of every question and, for a two-class
task whose prompt forecasts with one of its questions, which options forecast the positive class.
Random tables of that shape go through the functions a real table goes through, so the zero point
of cv_gain_pct is fitted on as many columns as the real one. That number of columns is what sets
how far below zero an answer without information scores.

Random answers come in two kinds, because a decider's answers either spread their probability
("soft") or sit near one option ("sharp"), and the two score differently by chance. A result is
placed among the kind its own answers resemble.

Off the training split the zero point also holds transfer_gain_pct. Its direction comes from V fitted
on other random answers of the same shape, drawn from a generator of their own so that the draws of
the other yardsticks stay as they were.

One file per split and signature. It is reused only while everything it depends on is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .layout import Layout, now, read_json, write_json_atomic
from .prompt import QuestionForecast, ReaderPrompt
from .questions import Space, Table
from .rows import Design
from .task import TaskConfig
from .yardsticks import binned_mi_pct, fit, forecast_rest, percentile, recalibrated_gain_pct, transfer_gain_pct, wauc

KINDS = ("soft", "sharp")
SHARP_WEIGHT = 0.9   # a result is placed among the sharp answers when this share of its weight is sharp
DIRECTION_STREAM = 1


@dataclass(frozen=True)
class Signature:
    spaces: tuple[Space, ...]
    forecast: QuestionForecast | None

    @classmethod
    def of(cls, prompt: ReaderPrompt, K: int) -> Signature:
        scalar = K == 2 and isinstance(prompt.forecast, QuestionForecast)
        return cls(tuple(q.space for q in prompt.questions), prompt.forecast if scalar else None)

    @property
    def text(self) -> str:
        """For file names: "p2-pos1" is one predicate whose yes forecasts the positive class,
        "c3_s5" a three-way choice and a five-level score without such a forecast."""
        parts = [s.token for s in self.spaces]
        if self.forecast:
            positive = (str(j) for j, c in enumerate(self.forecast.option_class) if c == 1)
            parts[self.forecast.question] += "-pos" + ".".join(positive)
        return "_".join(parts)


def _identity(task: TaskConfig, design: Design, sig: Signature) -> dict[str, Any]:
    """Everything a zero point depends on. A stored one is reused only when all of it matches."""
    m = task.metrics
    return {"split": design.split, "signature": sig.text, "reps": m.null_reps, "seed": m.null_seed, "rows": len(design.rows),
            "data_sha256": design.data_sha256, "sample": task.merged["train_sample"] if design.split == "train" else None,
            "classes": list(task.outcome.classes), "cv_groups": task.cv_groups, "cv_folds": m.cv_folds, "cv_reps": m.cv_reps,
            "mi_perms": m.mi_perms, **({"transfer": DIRECTION_STREAM} if design.split != "train" else {})}


def _summary(draws: list[float]) -> dict[str, float] | None:
    if not draws:
        return None
    a = np.asarray(draws, float)
    return {"median": round(float(np.median(a)), 3), "p05": round(float(np.percentile(a, 5)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "p99": round(float(np.percentile(a, 99)), 3)}


def _compute(task: TaskConfig, design: Design, sig: Signature, note: Callable[[str], None]) -> dict[str, Any]:
    """Repetition r draws from default_rng(seed + r): the soft answers of every question, then the
    sharp ones. For one predicate that is random(n), uniform(), random(n), as version 0.1.0 drew."""
    m, labels, n = task.metrics, design.labels, len(design.rows)
    transfer = design.split != "train"
    dist: dict[str, dict[str, list[float]]] = {
        kind: {"cv_gain_pct": [], **({"transfer_gain_pct": []} if transfer else {}), **({"auc": [], "mi_pct": []} if sig.forecast else {})}
        for kind in KINDS}
    for r in range(m.null_reps):
        rng = np.random.default_rng(m.null_seed + r)
        tables = {"soft": Table.random(sig.spaces, rng, n, sharp=False), "sharp": Table.random(sig.spaces, rng, n, sharp=True)}
        directions = np.random.default_rng((m.null_seed + r, DIRECTION_STREAM))
        for kind, table in tables.items():
            if kind == "sharp" and any((col == col[0]).all() for col in table.rest):
                continue
            if sig.forecast:
                s = forecast_rest(table.rest[sig.forecast.question], sig.forecast.option_class, 2)[:, 0]
                dist[kind]["mi_pct"].append(round(binned_mi_pct(s, labels, m.mi_perms)[0], 4))
                dist[kind]["auc"].append(round(wauc(s, labels.y, labels.w), 5))
            dist[kind]["cv_gain_pct"].append(round(recalibrated_gain_pct(table.features(), labels, m.cv_folds, m.cv_reps), 4))
            other = Table.random(sig.spaces, directions, n, sharp=kind == "sharp") if transfer else None
            if other is not None and not (kind == "sharp" and any((col == col[0]).all() for col in other.rest)):
                B = fit(other.features(), labels.z, labels.w, labels.K)
                dist[kind]["transfer_gain_pct"].append(round(transfer_gain_pct(table.features(), B, labels, m.cv_folds, m.cv_reps), 4))
        if (r + 1) % 50 == 0:
            note(f"  zero point on {design.split}: {r + 1}/{m.null_reps}")
    return {**_identity(task, design, sig), "time": now(), "dist": dist,
            "summary": {kind: {metric: _summary(draws) for metric, draws in dist[kind].items()} for kind in KINDS}}


def get(task: TaskConfig, layout: Layout, design: Design, sig: Signature, *, force: bool,
        note: Callable[[str], None]) -> dict[str, Any]:
    f = layout.null_file(design.split, sig.text)
    stored = read_json(f)
    if stored is not None and not force and all(stored.get(k) == v for k, v in _identity(task, design, sig).items()):
        return stored
    note(f"measuring the zero point on {design.split} for answers of shape {sig.text} "
         f"({task.metrics.null_reps} random answers of each kind; once per split and shape)...")
    zero = _compute(task, design, sig, note)
    write_json_atomic(f, zero)
    return zero


def place(zero: Mapping[str, Any], block: Mapping[str, Any], table: Table, w: np.ndarray) -> dict[str, Any]:
    """The percentiles of a result among the random answers of the kind its own answers resemble,
    and among both kinds for the record."""
    kind = "sharp" if w[table.sharp()].sum() / w.sum() >= SHARP_WEIGHT else "soft"
    among = {k: {metric: percentile(draws, block.get(metric)) for metric, draws in zero["dist"][k].items()} for k in KINDS}
    return {"kind": kind, **among[kind], "by_kind": among, "ref": zero["summary"][kind]}
