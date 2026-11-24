"""The free part of a study: what reads the training split and the logs only. Each function
returns what it found; show.py prints it.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from . import yardsticks as Y
from .answers import Spend, spent_usd
from .layout import AgentName, PlabError, write_json_atomic
from .expr import MAX_CHARS, MAX_NODES
from .prompt import MAX_QUESTIONS, MAX_RULES
from .questions import KINDS, machine_answer
from .rows import Design, breakeven, period_key
from .study import Candidate, Study
from .task import DEFAULTS, merge

AGREE_SUBSAMPLE_SEED = 4242

#: What `plab init` writes over DEFAULTS: a two-class task whose rows carry a value, used as a veto.
INIT_EXAMPLE: dict[str, Any] = {
    "rows": {"value": "value"},
    "outcome": {"meaning": {"1": "positive: the outcome a veto should avoid (for example a losing trade)",
                            "0": "negative (for example a winning trade)"}},
    "splits": {"train": ["2018-01-01", "2022-01-01"], "eval": ["2022-01-01", "2024-01-01"],
               "test": ["2024-01-01", "2026-01-01"]},
    "half_boundary": "2020-01-01",
    "train_sample": {"reference_per_other": 6, "seed": 12345},
    "decision": {"skip_if_prob_at_least": 0.5},
    "budget": {"usd": 2.0},
}

PREREG_TEMPLATE = """<!-- plab:template -->
# Pre-registration: {name}

Written on: YYYY-MM-DD, before anyone looked at the evaluation set.
(Delete the first line of this file when it is filled in; `plab prereg freeze` refuses the template.)

## Question
What is the outcome, what do the prompts ask a decision model about it, and what would count as success?

## Splits
Training / evaluation / test periods (they are in task.json; repeat them here in words).
Who may look at what: searchers see the training split; the evaluation split only through
`plab check` (capped per agent); the test split once, through `plab test --final`.

## Selection rule
Copy task.json "selection" here in words. If no candidate meets it, the test set stays unused.

## Decision models
Which deciders of task.json may be nominated (their kind and model are frozen with this file).

## Budget
Money, evaluation checks per agent, nominations per agent.

## What will be reported
The test result for the selected candidate, how many times the evaluation set was looked at, and the cost.
"""

EXAMPLE_PROMPT = {
    "name": "example: two veto rules",
    "fields": "all",
    "questions": [{"name": "skip", "type": "predicate",
                   "rules": [{"text": "(a) Skip when FIELD_A is greater than 10.", "expr": "FIELD_A > 10"},
                             {"text": "(b) Skip when FIELD_B is true.", "expr": "FIELD_B"}]}],
    "forecast": {"question": "skip", "classes": {"yes": "1", "no": "0"}},
}


def init(target: Path) -> Path:
    """Make a task folder: task.json, prereg.md, prompts/example.json, .gitignore. Never overwrites."""
    target.mkdir(parents=True, exist_ok=True)
    if (target / "task.json").exists():
        raise PlabError(f"{target / 'task.json'} already exists")
    name = target.resolve().name
    write_json_atomic(target / "task.json", {**merge(DEFAULTS, INIT_EXAMPLE), "name": name})
    (target / "prompts").mkdir(exist_ok=True)
    for path, text in ((target / "prereg.md", PREREG_TEMPLATE.format(name=name)),
                       (target / "prompts" / "example.json", json.dumps(EXAMPLE_PROMPT, indent=1) + "\n"),
                       (target / ".gitignore", ".plab/\nrows.jsonl\n")):
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    return target


def _summary(values: list[Any]) -> str | None:
    numbers = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if numbers:
        return "q0/5/25/50/75/95/100: " + " / ".join(f"{x:.4g}" for x in np.percentile(numbers, [0, 5, 25, 50, 75, 95, 100]))
    flags = [x for x in values if isinstance(x, bool)]
    if flags:
        return f"true {100 * sum(flags) / len(flags):.1f}%"
    texts = [x for x in values if isinstance(x, str)]
    if texts:
        count: dict[str, int] = {}
        for x in texts:
            count[x] = count.get(x, 0) + 1
        top = sorted(count.items(), key=lambda item: -item[1])[:4]
        return f"{len(count)} distinct; " + ", ".join(f"{x[:28]!r} {100 * c / len(texts):.0f}%" for x, c in top)
    sizes = [len(x) for x in values if isinstance(x, list)]
    return f"lists of {min(sizes)}-{max(sizes)} values" if sizes else None


def _auc(units: Design, name: str) -> dict[str, float | None]:
    """The weighted AUC of one numeric field alone, on the design and on each of its halves."""
    column = [r.state.get(name) for r in units.rows]
    if not all(x is None or isinstance(x, (bool, int, float)) for x in column):
        return {}
    col = np.array([np.nan if x is None else float(x) for x in column])
    y, w, half, has = units.labels.y, units.labels.w, units.labels.half, np.isfinite(col)
    if has.sum() <= 20:
        return {}
    parts = {"auc": has, "auc_h1": has & (half == 1), "auc_h2": has & (half == 2)}
    return {key: Y.rnd(Y.wauc(col[m], y[m], w[m]), 3) if m.any() else None for key, m in parts.items()}


def fields(study: Study, *, with_auc: bool) -> dict[str, Any]:
    """The training split at a glance, and each state field: its type, how often it is filled and
    a summary; with_auc (two classes) adds the AUC of each numeric field."""
    task, rows = study.task, study.rows("train").rows
    if with_auc and task.outcome.K != 2:
        raise PlabError("--auc ranks one class against the other: it needs two outcome classes")
    units = study.design("train") if with_auc else None
    listed = []
    for name in study.all_fields():
        values = [r.state.get(name) for r in rows]
        filled = [x for x in values if x is not None]
        listed.append({"field": name, "type": "/".join(sorted({type(x).__name__ for x in filled}) or ["none"]),
                       "filled_pct": round(100 * len(filled) / len(values), 1), "summary": _summary(filled),
                       **(_auc(units, name) if units else {})})
    counts = np.bincount([r.z for r in rows], minlength=task.outcome.K).tolist()
    valued = task.rows.value is not None
    return {"rows": len(rows), "class_counts": dict(zip(task.outcome.classes, counts)), "meaning": dict(task.outcome.meaning),
            "all_in_value_per_1000": 1000 * sum(r.value for r in rows) / len(rows) if valued else None,
            "breakeven": breakeven(rows) if valued and task.outcome.K == 2 else None, "fields": listed}


def _yes_at(study: Study) -> float:
    """Where a yes/no answer turns yes: the decision threshold, or one half without a decision."""
    return study.task.decision.skip_if_prob_at_least if study.task.decision else 0.5


def agree(study: Study, cand: Candidate, spend: Spend, *, per_rule: bool, limit: int | None) -> dict[str, Any]:
    """Does the decider answer like the machine? On the training design, per question the machine
    can answer (or per rule), how often the option the decider picks (`Kind.picked`: a yes/no
    answer is yes at the decision threshold) is the machine's. Units that either one refused are
    counted apart and left out. A question of two options also gets the four cells of yes and no;
    a longer one the most common disagreements. Logged."""
    if per_rule and not cand.asker.rules:
        raise PlabError("--per-rule asks the rules one at a time: this prompt has no rules")
    asker = cand.asker.per_rule(study.task.rule_frame) if per_rule else cand.asker
    if not any(q.machine_ready for q in asker.questions):
        raise PlabError("agree compares the decider with the machine: it needs a question the machine can answer "
                        "(rules with an expr on every rule, or a question with an expr)")
    units = study.design("train")
    if limit:
        units = units.take(np.sort(np.random.default_rng(AGREE_SUBSAMPLE_SEED).permutation(len(units.rows))[:limit]))
    a = study.answered(replace(cand, asker=asker), units, spend, partial_ok=True)
    threshold = _yes_at(study)
    data = [x.data for x in asker.asks(a.design.rows)]
    questions: dict[str, Any] = {}
    for q, rest, refused in zip(asker.questions, a.table.rest, a.table.refused):
        if not q.machine_ready:
            continue
        kind = KINDS[q.kind]
        machine = np.array([kind.chosen(machine_answer(q, d)) for d in data], int)
        both = ~refused & (machine >= 0)
        model, machine, p = kind.picked(rest, threshold)[both], machine[both], rest[both, 0]
        ids = [r.id for r, kept in zip(a.design.rows, both) if kept]
        differs = model != machine
        entry: dict[str, Any] = {"agree_pct": Y.rnd(100 * (~differs).mean(), 2) if both.any() else None,
                                 "model_refused": int(refused.sum()), "machine_refused": int((~refused & ~both).sum())}
        if len(q.options) == 2:
            yes, said_yes = model == 1, machine == 1
            entry.update({"both_skip": int((yes & said_yes).sum()), "model_only": int((yes & ~said_yes).sum()),
                          "machine_only": int((~yes & said_yes).sum()), "neither": int((~yes & ~said_yes).sum()),
                          "mean_answer_machine_hit": Y.rnd(p[said_yes].mean(), 3) if said_yes.any() else None,
                          "mean_answer_machine_miss": Y.rnd(p[~said_yes].mean(), 3) if (~said_yes).any() else None})
        else:
            pairs: dict[tuple[str, str], int] = {}
            for m_opt, d_opt in zip(machine[differs], model[differs]):
                pairs[(q.options[m_opt], q.options[d_opt])] = pairs.get((q.options[m_opt], q.options[d_opt]), 0) + 1
            entry["disagreements"] = [{"machine": k[0], "model": k[1], "n": n} for k, n in sorted(pairs.items(), key=lambda kv: -kv[1])[:6]]
        entry["disagreeing_ids"] = [i for i, d in zip(ids, differs) if d][:8]
        questions[q.name] = entry
    return study.layout.append_log(spend.agent, "agree", {
        "name": cand.asker.name, "prompt_id": cand.asker.prompt_id, "n": len(a.design.rows), "decider": cand.decider.name,
        "per_rule": per_rule, "threshold": threshold, "questions": questions, "calls": a.calls.calls, "cost_usd": a.calls.cost_usd})


def _drift_by_option(study: Study, cand: Candidate, a: Any, by: str) -> dict[str, Any]:
    """Per question and period of the training split: the share of units on each option the decider
    picks (`Kind.picked`), and the outcome among them. A refused unit is on no option."""
    task, labels, rows = study.task, a.design.labels, a.design.rows
    threshold = _yes_at(study)
    period = np.array([period_key(r.t, by) for r in rows])
    w, z, classes = labels.w, labels.z, task.outcome.classes

    def outcome(m: np.ndarray) -> Any:
        total = w[m].sum()
        if total <= 0:
            return None
        if labels.K == 2:
            return Y.rnd(100 * (w[m] * (z[m] == 1)).sum() / total, 2)
        return {classes[c]: Y.rnd(100 * (w[m] * (z[m] == c)).sum() / total, 1) for c in range(1, labels.K)}

    questions = {}
    for q, rest, refused in zip(cand.asker.questions, a.table.rest, a.table.refused):
        top = np.where(refused, -1, KINDS[q.kind].picked(rest, threshold))
        out = []
        for name in sorted(set(period)):
            m = period == name
            out.append({"period": name, "rows": int(m.sum()), "outcome": outcome(m),
                        "options": [{"share_pct": Y.rnd(100 * w[m & (top == j)].sum() / w[m].sum(), 1), "outcome": outcome(m & (top == j))}
                                    for j in range(len(q.options))]})
        questions[q.name] = {"options": list(q.options), "periods": out}
    return {"name": cand.asker.name, "by": by, "decider": cand.decider.name, "units": len(rows), "missing": a.missing,
            "threshold": threshold, "classes": list(classes), "questions": questions}


def drift(study: Study, cand: Candidate, spend: Spend, *, by: str) -> dict[str, Any]:
    """By period of the training split. With a decision in task.json and a prompt with a forecast,
    the veto use: how much is skipped, the positive rate of what is skipped and kept, the value, and
    how often each rule hits. Otherwise the answers of each question by option, and the outcome
    among them. A decider that stores nothing costs nothing and answers every training row; any
    other is read from its stored answers."""
    task = study.task
    a = study.answered(cand, study.design("train", sampled=cand.decider.stored), spend, partial_ok=True)
    if not task.decision or cand.asker.forecast is None:
        return _drift_by_option(study, cand, a, by)
    labels, rows = a.design.labels, a.design.rows
    skip = study.forecast(cand, a, None).s >= task.decision.skip_if_prob_at_least
    hits = cand.asker.rule_hits(rows)
    period = np.array([period_key(r.t, by) for r in rows])
    out = []
    for name in sorted(set(period)):
        m = period == name
        w, y, value, sk = labels.w[m], labels.y[m], labels.value[m], skip[m]
        total = w.sum()
        skipped = (w[sk] * y[sk]).sum() / w[sk].sum() if sk.any() else None
        kept = (w[~sk] * y[~sk]).sum() / w[~sk].sum() if (~sk).any() else None
        out.append({"period": name, "rows": int(m.sum()), "pos_pct": Y.rnd(100 * (w * y).sum() / total, 2),
                    "skip_pct": Y.rnd(100 * w[sk].sum() / total, 1),
                    "pos_pct_skipped": Y.rnd(100 * skipped, 2) if skipped is not None else None,
                    "pos_pct_kept": Y.rnd(100 * kept, 2) if kept is not None else None,
                    "lift": Y.rnd(skipped / kept, 2) if (skipped is not None and kept) else None,
                    "value_per_1000_all": Y.rnd(1000 * (w * value).sum() / total, 1),
                    "value_per_1000_rule": Y.rnd(1000 * (w[~sk] * value[~sk]).sum() / total, 1),
                    "rule_hit_pct": [None if h is None else Y.rnd(100 * w[h[m]].sum() / total, 1) for h in hits]})
    return {"name": cand.asker.name, "by": by, "decider": cand.decider.name, "units": len(rows), "missing": a.missing,
            "periods": out}


def board(study: Study, *, split: str, agent: AgentName | None, show_all: bool) -> list[dict[str, Any]]:
    """Training runs (the latest per prompt and decider, best cv_gain first), or one agent's own
    evaluation checks. Another agent's looks are shown only when the whole study is asked for."""
    if split == "train":
        latest = {(r["prompt_id"], r["decider"]): r for r in study.layout.read_log("runs", agent)}
        return sorted(latest.values(), key=lambda r: -r["cv_gain_pct"])
    if not agent and not show_all:
        raise PlabError("board --split eval shows one agent's own checks: give --agent NAME (or --all for the whole study)")
    return study.layout.read_log("checks", None if show_all else agent)


def cost(study: Study) -> dict[str, Any]:
    """What was spent, by decider, command and agent, and what is left of the budget."""
    by: dict[tuple[str, str, str], dict[str, float]] = {}
    for line in study.layout.read_log("ledger"):
        c = by.setdefault((line["decider"], line["cmd"], line["agent"]), {"calls": 0, "errors": 0, "usd": 0.0})
        c["calls"], c["errors"], c["usd"] = c["calls"] + line["calls"], c["errors"] + line["errors"], c["usd"] + float(line["cost_usd"])
    return {"spent_usd": spent_usd(study.layout), "budget_usd": study.task.budget.usd,
            "by": [{"decider": d, "cmd": cmd, "agent": a, **c} for (d, cmd, a), c in sorted(by.items())]}


def brief(study: Study, agent: AgentName, direction: str) -> str:
    """The brief for a searching agent: what it may read and run, its caps, and how to work."""
    task = study.task
    start, end = task.splits["train"]
    decision = task.decision
    use = (f"Use: a case is skipped when the forecast of class {task.outcome.classes[1]!r} is at least {decision.skip_if_prob_at_least}. "
           "Judge by value (the decision line, value per 1,000 units, next to taking everything), not by AUC alone.\n"
           if decision else "")
    return f"""# Brief for searching agent `{agent}`

Task folder: `{task.root}` (run every command from there, or pass `-C {task.root}`).
Goal: find prompts whose answers tell the most about the outcome on data the search never saw.
Outcome classes: {', '.join(task.outcome.classes)}. Meaning: {json.dumps(dict(task.outcome.meaning), ensure_ascii=False)}.
{use}Direction for you: {direction}

## What you may read
- `task.json`, `prereg.md`, your own prompt files under `prompts/{agent}/` (make the folder), your own logs under
  `.plab/agents/{agent}/`, and `docs/protocol.md` of the harness.
- The training split `.plab/data/train.jsonl` (from {start:%Y-%m-%d}, before {end:%Y-%m-%d}), rows and outcomes: look as much as you like.
- Outputs of the commands below.

## What you must not open
- `rows.jsonl`, `.plab/data/eval.jsonl`, `.plab/data/test.jsonl`, `.plab/answers/`, other agents' folders under
  `.plab/agents/`, `.plab/selected.json`, `.plab/test_result.json`. Of the evaluation split you see only the aggregate
  numbers `plab check` prints, never its rows, ids or outcomes. Do not subtract or combine the numbers of several checks
  to work out the counts or outcomes of a part of the evaluation split: that reads its content. The test split, never.

## Commands you may run (always with `--agent {agent}` where it is accepted)
- `plab fields [--auc]`: the fields of the state on the training split.
- `plab lint PROMPT`: wordings that make a model follow a rule less like the machine (multiplying by a number below 1,
  a maximum over a long list, 4 or more rules with "and"), and what the prompt leaves out.
- `plab run PROMPT --decider machine --agent {agent}`: the training split answered by the exprs alone. Free; start here.
- `plab run PROMPT --agent {agent}`: the training split with the task's decider (`{task.decider}`); a paid model costs money.
  Name several deciders (`--decider a --decider b`) to compare decision models on the same prompt.
- `plab agree PROMPT --agent {agent} [--decider NAME] [--per-rule] [--limit N]`: does the model answer like the machine, on every
  question (or rule) that has an expr?
- `plab drift PROMPT [--decider NAME] [--by year]`: by period of the training split, the answers of each question by option and the
  outcome among them (with a decision and a forecast: the skip share, rule hits and value). Look for what works in one era only.
- `plab survey [PROMPT] [--by half]`: per state field (or per rule and question of a prompt, as the machine answers), whether its
  slope holds across the periods of the training split, and what re-learning it every month gains. Free.
- `plab board --split train` and `plab board --split eval --agent {agent}`.
- `plab check PROMPT --agent {agent} [--decider NAME]`: ONE look at the evaluation split, as aggregate numbers only. You have
  {task.budget.eval_checks_per_agent} in total; each look is logged and reported, and every look makes the evaluation numbers
  read better than they are. Whether you may use them to steer the search or only to judge overfitting is in `prereg.md`.
- `plab nominate PROMPT --agent {agent} [--decider NAME]`: put forward at most {task.budget.nominations_per_agent} candidates.
  The harness, not you, selects by the pre-registered rule.
- A candidate is a prompt AND a decider. `check` and `nominate` use the task's decider (`{task.decider}`) unless you pass
  `--decider`, so pass the one you measured with (`--decider machine` for a prompt you ran with the machine).
- Never run: `split`, `prereg freeze`, `select`, `test`, `null --force`, `migrate`.

## Prompt format
```
{{"name": "two questions",
 "fields": ["FIELD_A", "FIELD_B"],
 "questions": [
  {{"name": "big", "type": "predicate", "instructions": "Is FIELD_A above 10?", "expr": "FIELD_A > 10"}},
  {{"name": "kind", "type": "choice", "instructions": "Which kind of case is this?",
   "choices": ["other", {{"value": "urgent", "description": "work cannot go on"}}], "expr": "('other', 'urgent')[FIELD_B]"}}],
 "forecast": {{"question": "big", "classes": {{"yes": {json.dumps(task.outcome.classes[-1])}, "no": {json.dumps(task.outcome.classes[0])}}}}}}}
```
- `fields`: `"all"`, a field group of task.json, or a list. The decider sees only these, and an expr may read only these.
- `questions`: 1 to {MAX_QUESTIONS}. `"type"` is `predicate` (with `instructions`, or `"rules"`: 1 to {MAX_RULES} of `{{"text", "expr"}}` or plain strings;
  it is yes when any rule applies), `choice` (`"choices"`: 2 to {KINDS['choice'].most}, each a string or `{{"value", "description"}}`) or `score`
  (`"levels"`: 2 to {KINDS['score'].most}, each a string or `{{"label", "description"}}`).
- `expr` (optional; what `--decider machine` answers): field names, constants, tuples and lists, comparisons (== != < <= > >= in,
  not in), and/or/not, + - * /, abs/min/max/len, subscripts and slices; at most {MAX_CHARS:,} characters and {MAX_NODES} nodes. A predicate's
  expr is read as true or false, a choice's must give an option's name, a score's a level index from 0. When a field it reads
  is None, or the computation fails, it has no value: a rule does not apply, a predicate answers no, a choice or a score is
  refused (so `== None` never holds).
- `forecast` (optional): one question and an outcome class for every option of it, or `"fitted"` (V fitted on the training
  answers of all questions). Without one, the prompt is measured by the information in its answers alone: cv_gain, and
  transfer_gain off the training split (no AUC, decision table or as-is gain).
- A key starting with `_` is a note.

## How to work
1. Write prompts in the format above (`prompts/` may hold examples). Give a rule or a question a machine `expr` when it can be
   written with the fields; a model then applies it almost exactly like the machine, so search with `--decider machine` for free
   and spend model calls on what needs words and on `plab agree`.
2. Read cv_gain next to its zero point (the percentile among random answers). A cv_gain below the 95th percentile of random
   answers is not evidence. cv_gain does not see which way an answer points; asis_gain, AUC and the decision line do.
   An evaluation check also shows transfer_gain: V kept as it was fitted on your training answers, so an answer whose
   relation to the outcome turned around since training scores at or below zero there, while cv_gain stays up.
3. Prefer prompts that work in both halves and in every period of `plab drift`. Rules tied to how the process is built
   (fixed time windows, scheduled events) survived regime changes in the case study; rules about recent trend or
   volatility level did not.
4. Report back: the prompts you tried (file, decider, train cv_gain and its percentile), how many evaluation checks you used
   ({task.budget.eval_checks_per_agent} at most), what you nominated and why, and what you think did not carry over.
"""
