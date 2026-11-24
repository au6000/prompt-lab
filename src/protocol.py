"""The guarded part of a study (docs/protocol.md): the frozen pre-registration, capped and logged
looks at the evaluation split, nominations, selection by the frozen rule, and the one test.

A study moves through phases. The phase is read off the files each time it is asked for, so
nothing can fall out of step after a crash:

    open  --freeze-->  frozen  --select-->  selected  --test-->  testing  -->  closed

PERMITS says which command may run in which phase. `enter` enforces it and returns a Lock: the
proof of a verified pre-registration, without which no evaluation or test row is read.
"""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from . import __version__, yardsticks as Y
from .answers import Spend
from .layout import SELECT, AgentName, Layout, PlabError, canonical, now, read_json, sha256_file, write_json_atomic
from .study import Candidate, Study
from .task import OPS, SPLITS

TEMPLATE_MARK = "<!-- plab:template -->"
PHASES = ("open", "frozen", "selected", "testing", "closed")
PERMITS: Mapping[str, tuple[str, ...]] = {
    "split": ("open",),
    "freeze": ("open", "frozen"),
    "null": ("frozen", "selected", "testing", "closed"),
    "check": ("frozen", "selected"),
    "nominate": ("frozen", "selected"),
    "select": ("frozen", "selected"),
    "test": ("selected", "testing"),
    "survey": ("closed",),
}
"""Commands that are not here read the training split or the logs only and run in every phase.
"null" is the zero point of the evaluation split. "survey" is the survey of every split (`plab survey
--splits all`); the survey of the training split alone reads no other row and runs in every phase."""

#: Commands whose reason for a refusal does not depend on the phase they are refused in.
_ONLY_WHEN = {
    "survey": "every split is open to a survey only once the test has run and the study is closed; until then "
              "`plab survey` reads the training split (leave out --splits all)",
}

_TOO_EARLY = {
    "open": "the pre-registration is not frozen: write prereg.md, then `plab prereg freeze` "
            "(evaluation checks, selection and the test are refused until then)",
    "frozen": "nothing selected yet (`plab select`)",
}
_TOO_LATE = {
    "frozen": "the pre-registration is frozen; the split cannot be redone (it is part of the lock)",
    "selected": "a selection was made under this pre-registration; the split and the lock cannot be redone",
    "testing": "a test run was started but did not finish: finish it with `plab test --final --resume` "
               "(nothing was shown, so it is still the one run)",
    "closed": "the test has been run; the study is closed (start a new study for new prompts)",
}
_BRIEF = ("auc", "auc_h1", "auc_h2", "mi_pct", "mi_p", "cv_gain_pct", "transfer_gain_pct", "asis_gain_pct", "decision", "null_pct", "xfit", "n",
          "class_counts", "refused")


@dataclass(frozen=True)
class Lock:
    """Proof that the pre-registration is frozen and that prereg.md, the locked settings and the
    split files are what was frozen. Only `verify` makes one."""
    record: Mapping[str, Any]


def phase(layout: Layout) -> str:
    test = read_json(layout.test_lock)
    if test is not None:
        return "closed" if test["status"] == "done" else "testing"
    if layout.selected.exists():
        return "selected"
    return "frozen" if layout.prereg_lock.exists() else "open"


def permit(layout: Layout, command: str) -> str:
    """Refuse a command the study's phase does not permit, with the reason. Returns the phase."""
    now_in, allowed = phase(layout), PERMITS[command]
    if now_in not in allowed:
        raise PlabError(_ONLY_WHEN.get(command) or (_TOO_EARLY if PHASES.index(now_in) < PHASES.index(allowed[0]) else _TOO_LATE)[now_in])
    return now_in


def _fingerprint(study: Study) -> dict[str, Any]:
    layout = study.layout
    if not layout.prereg_md.exists():
        raise PlabError("prereg.md not found (write down how you will choose before looking at the evaluation set)")
    data = {s: sha256_file(layout.split_file(s)) if layout.split_file(s).exists() else None for s in SPLITS}
    if data["train"] is None or data["eval"] is None:
        raise PlabError("split the data first (`plab split`)")
    return {"prereg_sha256": sha256_file(layout.prereg_md),
            "task_locked_sha256": hashlib.sha256(canonical(study.task.locked()).encode("utf-8")).hexdigest(), "data_sha256": data}


def freeze(study: Study, *, force: bool) -> dict[str, Any]:
    layout = study.layout
    frozen = permit(layout, "freeze") == "frozen"
    fingerprint = _fingerprint(study)
    text = layout.prereg_md.read_text(encoding="utf-8")
    if TEMPLATE_MARK in text:
        raise PlabError("prereg.md still carries the template mark; fill it in and delete the line " + TEMPLATE_MARK)
    if frozen and layout.read_log("checks"):
        raise PlabError("already frozen, and the evaluation set has been looked at: the lock cannot be moved")
    if frozen and not force:
        raise PlabError("already frozen (use --force to re-freeze; allowed only while no evaluation check exists)")
    record = {"time": now(), "harness_version": __version__, **fingerprint, "task_locked": study.task.locked()}
    write_json_atomic(layout.prereg_lock, record)
    layout.prereg_frozen.write_text(text, encoding="utf-8")
    return record


def verify(study: Study) -> Lock:
    """The Lock, or a refusal that names what changed since the pre-registration was frozen."""
    frozen = read_json(study.layout.prereg_lock)
    if frozen is None:
        raise PlabError(_TOO_EARLY["open"])
    current, changed = _fingerprint(study), []
    if current["prereg_sha256"] != frozen["prereg_sha256"]:
        changed.append("prereg.md")
    if current["task_locked_sha256"] != frozen["task_locked_sha256"]:
        old, new = frozen["task_locked"], study.task.locked()
        keys = [k for k in {**old, **new} if canonical(old.get(k)) != canonical(new.get(k))]
        changed.append("task.json (" + ", ".join(keys or ["settings"]) + ")")
    changed += [f".plab/data/{s}.jsonl" for s, h in current["data_sha256"].items() if h != frozen["data_sha256"].get(s)]
    if changed:
        raise PlabError("changed since the pre-registration was frozen: " + "; ".join(changed) + ". Refused. Put them back "
                        "as they were (prereg.frozen.md holds the frozen text) or start a new study")
    return Lock(frozen)


def enter(study: Study, command: str) -> Lock:
    permit(study.layout, command)
    return verify(study)


def lock_state(study: Study) -> str:
    frozen = read_json(study.layout.prereg_lock)
    if frozen is None:
        return "not frozen"
    try:
        verify(study)
    except PlabError as e:
        return f"frozen at {frozen['time']}; CHANGED: {e}"
    return f"frozen at {frozen['time']}; unchanged"


@contextmanager
def _one_check_at_a_time(layout: Layout, agent: AgentName) -> Iterator[None]:
    """Two checks of one agent running at once would both count the same looks and both pass the
    cap. The first to create this file runs; the other is refused."""
    path = layout.check_lock(agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise PlabError(f"another check by agent {agent} is running. If none is, one was killed midway: delete {path}")
    try:
        yield
    finally:
        path.unlink()


def check(study: Study, cand: Candidate, spend: Spend) -> dict[str, Any]:
    """One look at the evaluation split, counted against the agent's cap and logged."""
    lock = enter(study, "check")
    cap = study.task.budget.eval_checks_per_agent
    with _one_check_at_a_time(study.layout, spend.agent):
        used = len(study.layout.read_log("checks", spend.agent))
        if used >= cap:
            raise PlabError(f"agent {spend.agent} has used {used}/{cap} evaluation checks. Refused (every look at the "
                            "evaluation set is itself a chance to overfit it)")
        record = study.measure(cand, study.design("eval", lock), spend).record
        return study.layout.append_log(spend.agent, "checks", {"check_no": used + 1, "cap": cap, **record})


def nominate(study: Study, cand: Candidate, agent: AgentName) -> dict[str, Any]:
    """Put a candidate forward. The selection rule, not the agent, decides among the nominations."""
    enter(study, "nominate")
    cap = study.task.budget.nominations_per_agent
    mine = study.layout.read_log("nominations", agent)
    if any(r["prompt_id"] == cand.asker.prompt_id and r["decider"] == cand.decider.name for r in mine):
        raise PlabError("already nominated by this agent")
    if len(mine) >= cap:
        raise PlabError(f"agent {agent} has nominated {len(mine)}/{cap}. Refused")
    return study.layout.append_log(agent, "nominations", {
        "name": cand.asker.name, "prompt_id": cand.asker.prompt_id, "decider": cand.decider.name, "model": cand.decider.model,
        "file": cand.asker.file, "prompt": cand.asker.raw})


def _metric(measured: Mapping[str, Any], path: str) -> float | None:
    """"eval.decision.value_per_1000" -> measured["eval"]["decision"]["value_per_1000"]; None unless a number is there."""
    x: Any = measured
    for part in path.split("."):
        if not isinstance(x, dict) or part not in x:
            return None
        x = x[part]
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def failures(rule: Mapping[str, Any], measured: Mapping[str, Any]) -> list[str]:
    """The requirements of a selection rule that a candidate's measurements do not meet. A number
    that is missing fails its requirement."""
    values = [(req, _metric(measured, req["metric"])) for req in rule["require"]]
    return [f"{req['metric']} {req['op']} {req['value']} (is {v})" for req, v in values
            if v is None or not OPS[req["op"]](v, req["value"])]


def ranked(rule: Mapping[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Candidates in the order of the rule's rank_by keys; a missing number ranks last."""
    out = list(candidates)
    for key in reversed(rule["rank_by"]):
        sign = -1 if key["order"] == "desc" else 1
        out.sort(key=lambda c: (_metric(c["metrics"], key["metric"]) is None, sign * (_metric(c["metrics"], key["metric"]) or 0)))
    return out


def _brief(record: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: record[k] for k in _BRIEF if k in record}
    out["null_pct"] = {k: v for k, v in record["null_pct"].items() if k not in ("by_kind", "ref")}
    return out


def select(study: Study, spend: Spend) -> dict[str, Any]:
    """Measure every nomination on the training and evaluation splits and apply the frozen rule.
    The evaluation measurements are looks too, logged under the agent "select"."""
    lock = enter(study, "select")
    first: dict[tuple[str, str], dict[str, Any]] = {}
    for line in study.layout.read_log("nominations"):
        first.setdefault((line["prompt_id"], line["decider"]), line)
    if not first:
        raise PlabError("nothing has been nominated")
    rule, candidates = study.task.selection, []
    for line in first.values():
        cand = study.nominated(line)
        measured = {"train": study.measure(cand, study.design("train"), spend).record,
                    "eval": study.measure(cand, study.design("eval", lock), spend).record}
        study.layout.append_log(SELECT, "checks", measured["eval"])
        failed = failures(rule, measured)
        candidates.append({"name": line["name"], "prompt_id": line["prompt_id"], "decider": line["decider"],
                           "nominated_by": line["agent"], "passed": not failed, "failed": failed, "prompt": line["prompt"],
                           "metrics": measured})
    passing = ranked(rule, [c for c in candidates if c["passed"]])
    shown = ("name", "prompt_id", "decider", "nominated_by", "passed", "failed")
    out = {"time": now(), "rule": rule,
           "candidates": [{**{k: c[k] for k in shown}, "train": _brief(c["metrics"]["train"]), "eval": _brief(c["metrics"]["eval"])}
                          for c in candidates],
           "selected": {k: passing[0][k] for k in ("name", "prompt_id", "decider", "nominated_by", "prompt")} if passing else None,
           "reason": "first by rank_by among candidates that meet every requirement" if passing else
                     "no candidate meets every requirement: by the pre-registered rule the test set stays unused"}
    write_json_atomic(study.layout.selected, out)
    return out


def test_final(study: Study, spend: Spend, *, resume: bool) -> dict[str, Any]:
    """The selected candidate on the test split, once. A run that was started and did not finish
    showed nothing, so --resume may finish it for the same prompt."""
    lock = enter(study, "test")
    layout, task = study.layout, study.task
    chosen = read_json(layout.selected)["selected"]
    if chosen is None:
        raise PlabError("the selection found no prompt, so by the pre-registered rule the test set stays unused. Refused")
    if "test" not in task.splits:
        raise PlabError("task.json defines no test split")
    started = read_json(layout.test_lock)
    if started is not None and (started["prompt_id"] != chosen["prompt_id"] or not resume):
        raise PlabError(_TOO_LATE["testing"])
    cand = study.nominated(chosen)
    claim = {"status": "started", "started": started["started"] if started else now(),
             **{k: chosen[k] for k in ("prompt_id", "name", "decider")}}
    write_json_atomic(layout.test_lock, claim)
    m = study.measure(cand, study.design("test", lock), spend)
    labels = m.answered.design.labels
    result = {"time": now(), **{k: chosen[k] for k in ("name", "prompt_id", "decider")}}
    if task.decision and m.forecast is not None:
        threshold = task.decision.skip_if_prob_at_least
        result.update(threshold=threshold, selected=Y.final_summary(m.forecast.s, labels, threshold),
                      take_everything=Y.final_summary(np.zeros(len(labels.z)), labels, np.inf))
    result["info"] = {k: m.record[k] for k in ("n", "class_counts", "refused", "cv_gain_pct", "transfer_gain_pct", "asis_gain_pct",
                                               "auc", "null_pct")
                      if k in m.record}
    result.update(calls=m.record["calls"], cost_usd=m.record["cost_usd"])
    write_json_atomic(layout.test_result, result)
    write_json_atomic(layout.test_lock, {**claim, "status": "done", "finished": now()})
    return result


def report(study: Study) -> dict[str, Any]:
    layout, task = study.layout, study.task
    looks: dict[str, int] = {}
    for line in layout.read_log("checks"):
        looks[line["agent"]] = looks.get(line["agent"], 0) + 1
    cost: dict[str, dict[str, float]] = {}
    for line in layout.read_log("ledger"):
        c = cost.setdefault(line["decider"], {"calls": 0, "usd": 0.0})
        c["calls"], c["usd"] = c["calls"] + line["calls"], c["usd"] + float(line["cost_usd"])
    selection = read_json(layout.selected)
    return {"task": task.name, "phase": phase(layout), "prereg": lock_state(study),
            "eval_looks_total": sum(looks.values()), "eval_looks_by_agent": looks,
            "eval_checks_cap_per_agent": task.budget.eval_checks_per_agent, "train_runs": len(layout.read_log("runs")),
            "exploratory_surveys": len(layout.read_log("surveys")),
            "nominations": [{k: r[k] for k in ("agent", "name", "prompt_id", "decider")} for r in layout.read_log("nominations")],
            "selection_done": selection is not None,
            "selected": selection and selection["selected"] and {k: selection["selected"][k]
                                                                 for k in ("name", "prompt_id", "decider", "nominated_by")},
            "selection_reason": selection and selection["reason"], "test": read_json(layout.test_result),
            "cost": {d: {"calls": c["calls"], "usd": round(c["usd"], 6)} for d, c in cost.items()}, "budget_usd": task.budget.usd,
            "eval_zero_point": {f.name: read_json(f)["summary"] for f in layout.null_files("eval")}}
