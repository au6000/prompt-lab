"""The study: one task folder and the one pipeline every measuring command goes through.

    design (which units, in which order)  ->  asks (the prompt)  ->  answers (store, budget, decider)
    ->  table  ->  forecast  ->  yardsticks  ->  zero point  ->  record

What gets measured, nominated and selected is a Candidate: a prompt (what is read and asked) and a
decider (who answers). Tuning a prompt varies the first, comparing decision models the second.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from . import nulls, yardsticks as Y
from .answers import CallInfo, Spend, collect
from .deciders import Decider, make_decider
from .layout import Layout, PlabError
from .prompt import QuestionForecast, ReaderPrompt, load_prompt, parse_prompt
from .questions import Refused, Table
from .rows import Design, SplitRows, design, load_split, state_fields
from .task import TaskConfig, parse_task

if TYPE_CHECKING:
    from .protocol import Lock


@dataclass(frozen=True)
class Candidate:
    asker: ReaderPrompt
    decider: Decider


@dataclass(frozen=True)
class Answered:
    """The units that have answers, and the answers. `missing` counts the units left out because
    their ask failed; `refused` the units with a declined question, which stay in."""
    design: Design
    table: Table
    refused: int
    missing: int
    calls: CallInfo


@dataclass(frozen=True)
class Measurement:
    """The record as it is logged, with what it was computed from for callers that go on."""
    record: dict[str, Any]
    answered: Answered
    forecast: Y.Forecast | None


class Study:
    def __init__(self, root: str | Path, note: Callable[[str], None] = lambda text: None) -> None:
        """`note` receives progress lines (the zero point takes a while)."""
        self.task: TaskConfig = parse_task(Path(root))
        self.layout = Layout(self.task.root)
        self.note = note
        self._rows: dict[str, SplitRows] = {}

    def rows(self, split: str, lock: Lock | None = None) -> SplitRows:
        """The rows of a split. Training rows are free to read; evaluation and test rows take the
        Lock that only a verified pre-registration gives (protocol.enter)."""
        if split != "train" and lock is None:
            raise PlabError(f"the {split} rows are read only under a frozen pre-registration (`plab prereg freeze`), "
                            "through check, select and test")
        if split not in self._rows:
            self._rows[split] = load_split(self.task, self.layout, split)
        return self._rows[split]

    def design(self, split: str, lock: Lock | None = None, *, sampled: bool = True) -> Design:
        return design(self.task, self.rows(split, lock), sampled=sampled)

    def all_fields(self) -> tuple[str, ...]:
        return state_fields(self.task, self.rows("train"))

    def prompt(self, path: str) -> ReaderPrompt:
        return load_prompt(path, self.task, self.all_fields())

    def candidate(self, asker: ReaderPrompt, decider: str | None = None) -> Candidate:
        """A prompt with a decider of task.json, by name (the task's own decider when none is named)."""
        name = decider or self.task.decider
        if name not in self.task.deciders:
            raise PlabError(f"unknown decider {name!r} (task.json deciders has {', '.join(self.task.deciders)})")
        cand = Candidate(asker, make_decider(self.task.deciders[name], self.task.gateway))
        cand.decider.check(asker.questions)
        return cand

    def nominated(self, line: dict[str, Any]) -> Candidate:
        """The candidate a nomination or selection stored, rebuilt from its own copy of the prompt."""
        asker = parse_prompt(line["prompt"], self.task, self.all_fields(), name=line["name"])
        if asker.prompt_id != line["prompt_id"]:
            raise PlabError(f"nomination {line['name']}: the prompt no longer hashes to {line['prompt_id']} (task settings changed?)")
        return self.candidate(asker, line["decider"])

    def answered(self, cand: Candidate, units: Design, spend: Spend, *, partial_ok: bool = False) -> Answered:
        """The answers of a candidate for the units of a design."""
        answers, calls = collect(cand.asker.asks(units.rows), cand.decider, layout=self.layout, budget_usd=self.task.budget.usd,
                                 spend=spend, split=units.split, label={"name": cand.asker.name, "prompt_id": cand.asker.prompt_id})
        keep = [i for i, a in enumerate(answers) if a is not None]
        missing = len(answers) - len(keep)
        if missing and not partial_ok:
            raise PlabError(f"{missing} of {len(answers)} units have no answer ({calls.errors} errors this run). "
                            "Run the same command again to fill them; results are not shown on a partial set")
        got = [answers[i] for i in keep]
        return Answered(units.take(np.array(keep, int)) if missing else units, Table.of(cand.asker.questions, got),
                        sum(any(isinstance(x, Refused) for x in a) for a in got), missing, calls)

    def forecast(self, cand: Candidate, a: Answered, trained: np.ndarray | None) -> Y.Forecast | None:
        """The prompt's forecast on the answered units. `trained` is None on the training split, where
        a fitted forecast is out of fold; elsewhere it is V fitted on the training answers, and a
        fitted forecast is that V."""
        fc, labels, m = cand.asker.forecast, a.design.labels, self.task.metrics
        if isinstance(fc, QuestionForecast):
            return Y.Forecast(Y.forecast_rest(a.table.rest[fc.question], fc.option_class, labels.K), fitted=False)
        if fc is None:
            return None
        X = a.table.features()
        if trained is None:
            return Y.Forecast(Y.cv_predict(X, labels, m.cv_folds, m.cv_reps), fitted=True, out_of_fold=True)
        return Y.Forecast(Y.predict_rest(trained, X), fitted=True)

    def measure(self, cand: Candidate, units: Design, spend: Spend) -> Measurement:
        """One candidate on one design: its yardsticks, their place among random answers, its cost.
        Off the training split V is also fitted on the candidate's training answers (those not stored
        are asked for; a unit whose ask failed is left out, and the record counts it)."""
        a = self.answered(cand, units, spend)
        task, labels, m = self.task, a.design.labels, self.task.metrics
        train = None if units.split == "train" else self.answered(cand, self.design("train"), spend, partial_ok=True)
        trained = None if train is None else Y.fit(train.table.features(), train.design.labels.z, train.design.labels.w, labels.K)
        forecast = self.forecast(cand, a, trained)
        block = Y.score(a.table.features(), labels, cv_folds=m.cv_folds, cv_reps=m.cv_reps, mi_perms=m.mi_perms, forecast=forecast,
                        threshold=task.decision.skip_if_prob_at_least if task.decision else None, trained=trained)
        zero = nulls.get(task, self.layout, a.design, nulls.Signature.of(cand.asker, labels.K), force=False, note=self.note)
        counts = np.bincount(labels.z, minlength=labels.K).tolist()
        record = {"name": cand.asker.name, "prompt_id": cand.asker.prompt_id, "decider": cand.decider.name,
                  "model": cand.decider.model, "file": cand.asker.file, "split": units.split, "n": len(a.design.rows),
                  "class_counts": dict(zip(task.outcome.classes, counts)), "missing": a.missing, "refused": a.refused, **block}
        if forecast is not None and forecast.fitted:
            B = trained if trained is not None else Y.fit(a.table.features(), labels.z, labels.w, labels.K)
            record["lr_coef"] = [[round(float(x), 4) for x in row] for row in B]
        if train is not None:
            record["trained_on"] = {"units": len(train.design.rows), "missing": train.missing}
        record["null_pct"] = nulls.place(zero, block, a.table, labels.w)
        record.update(calls=a.calls.calls, cost_usd=a.calls.cost_usd, errors=a.calls.errors)
        if a.calls.via:
            record["answers_via"] = dict(a.calls.via)
        if units.split == "train" and labels.K == 2 and any(r.expr for r in cand.asker.rules):
            record["rule_hits"] = Y.rule_table(cand.asker.rule_hits(a.design.rows), labels)
        return Measurement(record, a, forecast)
