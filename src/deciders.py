"""Deciders: who answers. One class per kind in task.json `deciders`; a decider is picked by name
(--decider NAME) and no other module branches on its kind.

  machine  the exprs of the questions, evaluated in Python. Free and exact; nothing is stored.
  fake     fixed pseudo-answers for tests and demos: the machine's answer with noise, or noise alone
           where a question has no expr. Stored and entered in the ledger like a paid decider.
  gateway  a decision model on the Vercel AI Gateway (gateway.py).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Mapping, Sequence

from .answers import Reply
from .gateway import Client, sent
from .layout import PlabError, canonical, short_hash
from .prompt import Ask
from .questions import KINDS as QUESTION_KINDS, Question, fake_answer, machine_answer
from .task import DeciderSpec, GatewaySettings


@dataclass(frozen=True)
class Decider:
    """`name` is the key in task.json, `model` what records and the store name. The answers of a
    `stored` decider go through the store, the budget and the ledger; only such a decider is asked
    what it reads and what it would cost."""
    name: str
    model: str
    stored: ClassVar[bool] = True

    def check(self, questions: Sequence[Question]) -> None:
        """Refuse questions this decider cannot answer, before anything is asked."""

    def reads(self, ask: Ask) -> Mapping[str, Any]:
        """Everything about an ask that can change this decider's answer. The store keys on it."""
        raise NotImplementedError

    def estimate_usd(self, asks: Sequence[Ask]) -> float:
        raise NotImplementedError

    def answer(self, asks: Sequence[Ask]) -> list[Reply]:
        """One reply per ask, in order."""
        raise NotImplementedError


@dataclass(frozen=True)
class MachineDecider(Decider):
    stored: ClassVar[bool] = False

    def check(self, questions):
        missing = [f"rule {r.number}" for q in questions for r in q.rules if r.expr is None]
        missing += [f"question {q.name}" for q in questions if not q.rules and q.expr is None]
        if missing:
            raise PlabError(f"the machine decider needs an expr on every rule and question (missing: {', '.join(missing)})")

    def answer(self, asks):
        return [Reply(tuple(machine_answer(q, a.data) for q in a.questions)) for a in asks]


def _uniform(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def _fake_seed(ask: Ask) -> str:
    """The answer-cache key of version 0.1.0 for the fake backend, rebuilt from an ask. The fake's
    answers were seeded with it, so they stay what they were for every prompt 0.1.0 could hold.
    An expr outside a rule (new since) joins the seed only where there is one."""
    plain = {q.name: q.expr.src for q in ask.questions if q.expr}
    return short_hash({"questions": {q.name: QUESTION_KINDS[q.kind].seed(q) for q in ask.questions}, "fields": list(ask.data),
                       "state_extra": None, "model": "fake", "state_format": "json-object-v1",
                       "exprs": [r.expr and r.expr.src for q in ask.questions for r in q.rules],
                       **({"plain_exprs": plain} if plain else {})})


@dataclass(frozen=True)
class FakeDecider(Decider):
    cost_per_call: float

    def reads(self, ask):
        return {**sent(ask), "unit": ask.unit, "seed": _fake_seed(ask)}

    def estimate_usd(self, asks):
        return len(asks) * self.cost_per_call

    def answer(self, asks):
        replies = []
        for a in asks:
            seed = _fake_seed(a)
            answers = tuple(fake_answer(q, partial(_uniform, seed, a.unit, q.name), a.data) for q in a.questions)
            replies.append(Reply(answers, cost_usd=self.cost_per_call))
        return replies


@dataclass(frozen=True)
class GatewayDecider(Decider):
    price_per_mtok: float
    settings: GatewaySettings

    def reads(self, ask):
        """The request, and the order of its fields: the store key hashes with sorted keys."""
        return {**sent(ask), "fields": list(ask.data)}

    def estimate_usd(self, asks):
        """About 250 tokens of overhead and one token per three characters of the request."""
        return sum(250 + len(canonical(sent(a))) / 3 for a in asks) * self.price_per_mtok / 1e6

    def answer(self, asks):
        return Client(self.model, self.settings).decide_many(asks)


KINDS: Mapping[str, Callable[[DeciderSpec, GatewaySettings], Decider]] = {
    "machine": lambda s, g: MachineDecider(s.name, "machine"),
    "fake": lambda s, g: FakeDecider(s.name, "fake", s.settings["cost_per_call"]),
    "gateway": lambda s, g: GatewayDecider(s.name, s.settings["model"], s.settings["price_per_mtok"], g),
}


def make_decider(spec: DeciderSpec, gateway: GatewaySettings) -> Decider:
    return KINDS[spec.kind](spec, gateway)
