"""`plab migrate`: turn a task folder of prompt-lab 0.1 into one of 0.2.

task.json is rewritten from the 0.1 settings merged with 0.1's own defaults, so every value 0.1
used is written out and nothing leans on today's defaults. Prompt files stay as they are (0.2
reads the 0.1 forms), and so do the split files.

The answers 0.1 paid for move into the store under the keys the gateway decider computes for the
same request, marked via "evaluate": a predicate's probability, and a choice's probabilities when
0.1 kept one for every option. A score's answers stay behind, since 0.1 kept only
score/(levels - 1), and so does an answer whose unit is in no split file. The 0.1 ledger becomes
the ledger of the agent "migrate", so what 0.1 spent still counts against the budget. The 0.1
cache, ledger, logs and zero points are then set aside under .plab/archive-v0.1/: their fields
and prompt ids are 0.1's.

A study frozen under 0.1 is refused unless --unfreeze is given, which also sets aside its lock,
checks, nominations and selection, so the study starts again unfrozen. A study whose test was run
is finished and is never unfrozen.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .answers import Reply, keep
from .deciders import GatewayDecider
from .layout import MIGRATE, Layout, PlabError, append_jsonl, read_json, read_jsonl, write_json_atomic
from .prompt import Ask, v01_questions
from .questions import KINDS
from .task import FORMAT, SPLITS, GatewaySettings, merge

V01_DEFAULTS: dict[str, Any] = {
    "name": "task",
    "rows": "rows.jsonl",
    "fields": {"id": "id", "time": "t", "label": "label", "value": "value", "state": "state"},
    "label_meaning": {"1": "positive: the outcome a veto should avoid", "0": "negative"},
    "splits": None,
    "cv_groups": "month",
    "half_boundary": None,
    "train_sample": {"neg_per_pos": 6, "seed": 12345},
    "decision": {"skip_if_prob_at_least": 0.5},
    "state_fields": "all",
    "field_groups": {},
    "veto_head": ("These are veto rules (reasons not to act). If at least one of them applies to this case, the case is skipped; "
                  "if none of them applies, the case is taken. Apply the rules to the values given. "),
    "veto_tail": " Following these rules, should this case be skipped?",
    "selection": {
        "require": [{"metric": "train.cv_gain_pct", "op": ">", "value": 0},
                    {"metric": "eval.cv_gain_pct", "op": ">", "value": 0},
                    {"metric": "eval.decision.value_per_1000", "op": ">", "value": 0}],
        "rank_by": [{"metric": "eval.decision.value_per_1000", "order": "desc"},
                    {"metric": "eval.decision.skip_frac", "order": "asc"}],
    },
    "budget": {"usd": 5.0, "eval_checks_per_agent": 10, "nominations_per_agent": 2},
    "backend": {"name": "jev", "model": "typesafe-ai/jev", "workers": 6, "price_per_mtok": 0.042, "fake_cost_per_call": 0.0,
                "retries": 4},
    "metrics": {"cv_folds": 10, "cv_reps": 3, "mi_perms": 500, "null_reps": 200, "null_seed": 777},
}
V01_PAID = "jev"   # the one paid backend of 0.1; task_v1_to_v2 names the gateway decider after it
V01_LOGS = ("runs.jsonl", "agree.jsonl", "null_train.json", "null_eval.json")
V01_FROZEN = ("prereg.lock", "prereg.frozen.md", "checks.jsonl", "nominations.jsonl", "selected.json")
V01_SPENT = ("cache", "ledger.jsonl")
ARCHIVE = "archive-v0.1"


def _metric(path: str) -> str:
    """A record's count of positive units is now its count of class 1."""
    return path[:-len("positives")] + "class_counts.1" if path.endswith(".positives") else path


def task_v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    """The 0.2 task.json that measures what this 0.1 task.json measured."""
    notes = {k: v for k, v in raw.items() if k.startswith("_")}
    given = {k: v for k, v in raw.items() if not k.startswith("_")}
    c = {**merge(V01_DEFAULTS, given), "label_meaning": given.get("label_meaning", V01_DEFAULTS["label_meaning"])}
    where, backend, sample, rule = c["fields"], c["backend"], c["train_sample"], c["selection"]
    selection = {"require": [{**r, "metric": _metric(r["metric"])} for r in rule.get("require", [])],
                 "rank_by": [{"metric": _metric(k["metric"]), "order": k.get("order", "desc")} for k in rule.get("rank_by", [])]}
    return {
        "format": FORMAT, "name": c["name"], **notes,
        "rows": {"file": c["rows"], "id": where["id"], "time": where["time"], "outcome": where["label"], "value": where["value"],
                 "state": where["state"]},
        "outcome": {"classes": ["0", "1"], "meaning": c["label_meaning"]},
        "splits": c["splits"], "cv_groups": c["cv_groups"], "half_boundary": c["half_boundary"],
        "train_sample": ({"reference_per_other": sample["neg_per_pos"], "seed": sample.get("seed", 12345)}
                         if sample and sample.get("neg_per_pos") else None),
        "decision": c["decision"], "state_fields": c["state_fields"] or "all", "field_groups": c["field_groups"],
        "rule_frame": {"head": c["veto_head"], "tail": c["veto_tail"]},
        "selection": selection, "budget": c["budget"],
        "deciders": {"jev": {"kind": "gateway", "model": backend["model"], "price_per_mtok": backend["price_per_mtok"]},
                     "fake": {"kind": "fake", "cost_per_call": backend["fake_cost_per_call"]}},
        "decider": "machine" if backend["name"] == "mechanical" else backend["name"],
        "gateway": {"workers": backend["workers"], "retries": backend["retries"], "timeout_s": 120},
        "metrics": c["metrics"],
    }


def _v01_answers(layout: Layout, v2: dict[str, Any]) -> tuple[list[tuple[GatewayDecider, list[Ask], list[Reply]]], list[str]]:
    """The answers of the 0.1 cache that move into the store, shelf by shelf, and a line on each shelf."""
    metas = sorted((layout.plab / "cache" / V01_PAID).glob("*.meta.json"))
    states = {r["id"]: r["state"] for split in SPLITS for r in read_jsonl(layout.split_file(split))} if metas else {}
    spec, settings = v2["deciders"][V01_PAID], GatewaySettings(**v2["gateway"])
    moves, lines = [], []
    for meta_file in metas:
        key = meta_file.name[:-len(".meta.json")]
        cached = [r for r in read_jsonl(meta_file.with_name(f"{key}.jsonl")) if "a" in r]
        if not cached:
            continue
        where, meta = f".plab/cache/{V01_PAID}/{key}", read_json(meta_file)
        try:
            if meta.get("state_extra"):
                raise PlabError("its prompt gave state_extra, which a prompt of 0.2 cannot give")
            questions = v01_questions(meta["questions"])
            asks, replies = [], []
            for r in cached:
                answers = tuple(KINDS[q.kind].v01(r["a"].get(q.name), q) for q in questions)
                if r["id"] in states and None not in answers:
                    asks.append(Ask(r["id"], {f: states[r["id"]].get(f) for f in meta["fields"]}, questions))
                    replies.append(Reply(answers))
        except PlabError as e:
            lines.append(f"{where}: not moved: {e}")
            continue
        if asks:
            moves.append((GatewayDecider(V01_PAID, meta["model"], spec["price_per_mtok"], settings), asks, replies))
        left = len(cached) - len(asks)
        lines.append(f"{where}: {len(asks)} answers move into the store, marked via evaluate"
                     + (f"; {left} stay behind (the unit is in no split file, or a choice lacks an option's probability)" if left else ""))
    return moves, lines


def migrate(root: Path, *, write: bool, unfreeze: bool) -> list[str]:
    """What the migration does, line by line. Without `write` nothing is changed. task.json is
    written last, so a migration that stops half way is done again from the start."""
    layout = Layout(root.resolve())
    if not layout.task_json.exists():
        raise PlabError(f"no task.json in {layout.root}")
    try:
        raw = json.loads(layout.task_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PlabError(f"task.json is not valid JSON: {e}")
    if isinstance(raw, dict) and raw.get("format") == FORMAT:
        return [f"task.json is already in format {FORMAT}: nothing to do"]
    frozen = (layout.plab / "prereg.lock").exists()
    if frozen and layout.test_lock.exists():
        raise PlabError("this study started its test under prompt-lab 0.1. Its test set is used; it is not migrated")
    if frozen and not unfreeze:
        raise PlabError("this study was frozen under prompt-lab 0.1, and 0.2 measures with a changed instrument. Finish it with 0.1, or "
                        f"pass --unfreeze: the lock, the checks, the nominations and the selection are set aside under .plab/{ARCHIVE}/ "
                        "and the study starts again unfrozen")
    v2 = task_v1_to_v2(raw)
    moves, lines = _v01_answers(layout, v2)
    spent = read_jsonl(layout.plab / "ledger.jsonl")
    set_aside = [name for name in (*V01_LOGS, *V01_FROZEN, *V01_SPENT) if (layout.plab / name).exists()]
    lines = [f"task.json: prompt-lab 0.1 -> format {FORMAT} (the 0.1 file is kept as task.v1.json)",
             "prompt files and split files: unchanged (0.2 reads them as they are)", *lines]
    if spent:
        lines.append(f"the 0.1 ledger (${sum(float(r.get('cost_usd') or 0) for r in spent):.4f} in {len(spent)} lines) "
                     f"becomes the ledger of agent {MIGRATE}, so it counts against the budget")
    if set_aside:
        lines.append(f"set aside under .plab/{ARCHIVE}/: {', '.join(set_aside)}")
    if frozen:
        lines.append("the study is unfrozen: freeze it again with `plab prereg freeze` before any evaluation check")
    if not write:
        return [*lines, "nothing was changed: add --write to do it"]
    for decider, asks, replies in moves:
        keep(layout, decider, MIGRATE, asks, replies, via="evaluate")
    if spent:
        layout.log(MIGRATE, "ledger").unlink(missing_ok=True)
        append_jsonl(layout.log(MIGRATE, "ledger"), [{"decider" if k == "backend" else k: v for k, v in r.items()} for r in spent])
    for name in set_aside:
        (layout.plab / ARCHIVE).mkdir(exist_ok=True)
        (layout.plab / name).replace(layout.plab / ARCHIVE / name)
    (layout.root / "task.v1.json").write_bytes(layout.task_json.read_bytes())
    write_json_atomic(layout.task_json, v2)
    return lines
