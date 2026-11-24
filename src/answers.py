"""Answers: what a decider replied, where the replies are kept, and what they cost.

The store holds every answer that was paid for, keyed by what its decider read:

    .plab/answers/<shelf>/<agent>.jsonl    shelf = hash(model, questions as asked)
    {"in": hash(decider.reads(ask)), "id": unit, "answers": [...], "cost_usd": ..., "time": ..., "via"?: "evaluate"}

`via` marks an answer that version 0.1 fetched through the older /v1/evaluate (plab migrate).

A gateway model reads the input and the questions, so editing an expr or a forecast reuses its
answers. The fake decider reads the exprs too, so its key changes with them. Lines are appended
per agent and per chunk: an interrupted run resumes, and two agents never write one file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from .layout import AgentName, Layout, PlabError, append_jsonl, now, read_jsonl, short_hash, write_json_atomic
from .prompt import Ask
from .questions import Answer, Question, from_cell, to_cell, wire

if TYPE_CHECKING:
    from .deciders import Decider

CHUNK = 60   # asks per decider call and per ledger line


@dataclass(frozen=True)
class Reply:
    """What came back for one ask: its answers in question order, or None and the error that left
    it unanswered (it is asked again on the next run). After a `stop` reply no call can succeed (no
    credit, a refused key, answers to other questions): the run stops once its chunk is stored."""
    answers: tuple[Answer, ...] | None
    error: str = ""
    cost_usd: float = 0.0
    stop: bool = False


@dataclass(frozen=True)
class Spend:
    """What one command may spend: who asks, under which command, whether the decider may be called
    at all, and whether the budget check may be passed over (--yes)."""
    cmd: str
    agent: AgentName
    cache_only: bool = False
    over_budget_ok: bool = False


@dataclass(frozen=True)
class CallInfo:
    """The calls one collect made. `via` counts the answers by the endpoint that gave them, when
    they came from more than one: two instruments in one table."""
    calls: int
    errors: int
    cost_usd: float
    via: Mapping[str, int]


def spent_usd(layout: Layout) -> float:
    return sum(float(r.get("cost_usd") or 0) for r in layout.read_log("ledger"))


def _stored(layout: Layout, shelf: str, questions: Sequence[Question]) -> tuple[dict[str, tuple[Answer, ...]], set[str]]:
    """The last good line per key across every agent's part of a shelf, and the keys answered
    through /v1/evaluate."""
    lines = [r for part in sorted(layout.shelf(shelf).glob("*.jsonl")) for r in read_jsonl(part) if "answers" in r]
    return ({r["in"]: tuple(from_cell(q, c) for q, c in zip(questions, r["answers"])) for r in lines},
            {r["in"] for r in lines if r.get("via") == "evaluate"})


def shelf_key(decider: Decider, questions: Sequence[Question]) -> str:
    return short_hash({"model": decider.model, "questions": [wire(q) for q in questions]})


def keep(layout: Layout, decider: Decider, agent: AgentName, asks: Sequence[Ask], replies: Sequence[Reply], *,
         via: str | None = None) -> None:
    """Append the replies to asks of one prompt to the store, under the keys of the decider."""
    questions = asks[0].questions
    shelf = layout.shelf(shelf_key(decider, questions))
    if not (shelf / "meta.json").exists():
        write_json_atomic(shelf / "meta.json", {"model": decider.model, "questions": [wire(q) for q in questions], "created": now()})
    time = now()
    append_jsonl(shelf / f"{agent}.jsonl",
                 [{"in": short_hash(decider.reads(a)), "id": a.unit, "error": r.error[:300]} if r.answers is None else
                  {"in": short_hash(decider.reads(a)), "id": a.unit, "answers": [to_cell(q, x) for q, x in zip(questions, r.answers)],
                   "cost_usd": r.cost_usd, "time": time, **({"via": via} if via else {})} for a, r in zip(asks, replies)])


def collect(asks: Sequence[Ask], decider: Decider, *, layout: Layout, budget_usd: float, spend: Spend, split: str,
            label: dict[str, str]) -> tuple[list[tuple[Answer, ...] | None], CallInfo]:
    """The answers to every ask, from the store or the decider; None where an ask failed.

    What is not stored is asked once per key (units a decider cannot tell apart share an answer),
    in chunks. Each chunk is stored and entered in the ledger before the next is asked, so what was
    paid for is never lost.
    """
    if not decider.stored:
        return [r.answers for r in decider.answer(asks)], CallInfo(0, 0, 0.0, {})
    questions = asks[0].questions
    shelf = shelf_key(decider, questions)
    keys = [short_hash(decider.reads(a)) for a in asks]
    have, evaluated = _stored(layout, shelf, questions)
    first_with_key: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k not in have:
            first_with_key.setdefault(k, i)
    todo = list(first_with_key.values())
    if todo and spend.cache_only:
        raise PlabError(f"{len(todo)} of {len(asks)} answers are not stored and --cache-only was given (no calls made)")
    if todo:
        estimate = decider.estimate_usd([asks[i] for i in todo])
        spent = spent_usd(layout)
        if estimate > budget_usd - spent + 1e-12 and not spend.over_budget_ok:
            raise PlabError(f"{decider.name}: {len(todo)} calls, estimated ${estimate:.4f}, exceed the remaining budget "
                            f"${max(budget_usd - spent, 0):.4f} (spent ${spent:.4f} of ${budget_usd}). Raise budget.usd or pass --yes")
    calls = errors = 0
    cost = 0.0
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        replies = decider.answer([asks[i] for i in chunk])
        keep(layout, decider, spend.agent, [asks[i] for i in chunk], replies)
        failed = sum(r.answers is None for r in replies)
        paid = sum(r.cost_usd for r in replies)
        layout.append_log(spend.agent, "ledger", {"cmd": spend.cmd, "split": split, **label, "shelf": shelf, "decider": decider.name,
                                                  "model": decider.model, "calls": len(chunk), "errors": failed,
                                                  "cost_usd": round(paid, 8)})
        have.update({keys[i]: r.answers for i, r in zip(chunk, replies) if r.answers is not None})
        calls, errors, cost = calls + len(chunk), errors + failed, cost + paid
        stop = next((r.error for r in replies if r.stop), None)
        if stop:
            raise PlabError(f"{decider.name}: {stop}. The run stopped; the answers paid for so far are stored")
    old = sum(k in evaluated for k in keys if k in have)
    new = sum(k in have for k in keys) - old
    via = {"/v1/evaluate": old, "/v1/decisions": new} if old and new else {}
    return [have.get(k) for k in keys], CallInfo(calls, errors, round(cost, 8), via)
