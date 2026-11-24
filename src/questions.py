"""Typed questions and answers. Everything that depends on the kind of a question (predicate,
choice, score) is in its entry of KINDS: how its options are written in a prompt file and sent on
the wire, how an answer is stored, how answers become what V reads, how random answers are drawn
for the zero point, what the machine and the fake decider answer, and which columns `plab survey`
reads. No other module branches on a kind; a new kind is one class here.

Option 0 of a question is its reference option: "no" of a predicate, the first choice, the lowest
level. The answers of many units to one question are held in reference form (see yardsticks.py):
an (N, options - 1) array of the probabilities of the options after the first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import numpy as np

from .expr import Expr
from .layout import PlabError
from .yardsticks import log_ratios

KindName = Literal["predicate", "choice", "score"]
Uniform = Callable[..., float]

# The first two are part of the numbers on record: version 0.1.0 drew and judged its zero point with them.
SHARP_YES_SHARE = (0.05, 0.6)   # a sharp random predicate says yes on a share of the units drawn from this range
SHARP_WITHIN = 0.15             # an answer is sharp when at most this much probability lies off its top option
FAKE_LEAN = 0.9                 # what the fake decider puts on the machine's option of a choice or a score


@dataclass(frozen=True)
class Rule:
    """One clause of a rules predicate. `number` counts from 1 across the whole prompt."""
    number: int
    text: str
    expr: Expr | None


@dataclass(frozen=True)
class Question:
    """A question as a decider is asked it.

    A predicate may be made of rules: its instructions are then the task's rule frame around the
    rule texts, and the machine says yes when any rule applies. Any other question may carry one
    expr as its machine form: truthy for a predicate, an option for a choice, a level index for a score.
    """
    name: str
    kind: KindName
    instructions: str
    options: tuple[str, ...]
    descriptions: tuple[str | None, ...]
    rules: tuple[Rule, ...]
    expr: Expr | None

    @property
    def space(self) -> Space:
        return Space(self.kind, len(self.options))

    @property
    def machine_ready(self) -> bool:
        return all(r.expr for r in self.rules) if self.rules else self.expr is not None

    def machine_value(self, data: Mapping[str, Any]) -> Any:
        if self.rules:
            return any(r.expr.applies(data) for r in self.rules)
        return self.expr.value(data)


@dataclass(frozen=True)
class Yes:
    """A predicate's answer: the probability of yes."""
    p: float


@dataclass(frozen=True)
class Chose:
    """A choice's answer: the index of the option chosen, and the probability of every option when
    the decider gave them."""
    probs: tuple[float, ...] | None
    point: int


@dataclass(frozen=True)
class Scored:
    """A score's answer: the mean level index, and the probability of every level when the decider
    gave them."""
    probs: tuple[float, ...] | None
    point: float


@dataclass(frozen=True)
class Refused:
    """The decider declined the question. It is measured as the uniform answer and counted: it
    carries no information, and dropping its unit would change who is measured."""


Answer = Yes | Chose | Scored | Refused


@dataclass(frozen=True)
class Space:
    """The answer space of a question: its kind and number of options. What a zero point matches."""
    kind: KindName
    n: int

    @property
    def token(self) -> str:
        return f"{KINDS[self.kind].token}{self.n}"


class Kind(Protocol):
    token: str
    key: str | None
    takes_rules: bool

    def options(self, raw: Mapping[str, Any], where: str) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
        """The options and their descriptions, from a question of a prompt file."""

    def wire(self, q: Question) -> dict[str, Any]:
        """What the question sends beside its type, name and instructions."""

    def cell(self, a: Any, q: Question) -> dict[str, Any]:
        """The stored form of an answer."""

    def answer(self, cell: Mapping[str, Any], q: Question) -> Answer:
        """An answer from its stored form."""

    def parse(self, item: Mapping[str, Any], q: Question) -> Answer:
        """An answer from the Decisions API (one entry of its "answers", not a refusal). ValueError
        when it is not an answer of this kind."""

    def v01(self, value: Any, q: Question) -> Answer | None:
        """An answer from what version 0.1 cached; None when the value holds less than an answer."""

    def row(self, a: Any, n: int) -> list[float]:
        """One answer in reference form."""

    def features(self, rest: np.ndarray) -> np.ndarray:
        """What V reads of this question, (N, d)."""

    def soft(self, n: int, rng: np.random.Generator, size: int) -> np.ndarray:
        """Random answers that spread their probability, in reference form."""

    def sharp(self, n: int, rng: np.random.Generator, size: int) -> np.ndarray:
        """Random answers that each put all their probability on one option, in reference form."""

    def machine(self, q: Question, value: Any) -> Answer:
        """The answer the value of the question's machine form stands for."""

    def picked(self, rest: np.ndarray, threshold: float) -> np.ndarray:
        """Per unit, the option an answer in reference form picks."""

    def chosen(self, a: Answer) -> int:
        """The option one answer picks; -1 for a refusal."""

    def factors(self, q: Question, picked: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """The survey's columns of a question, (label, column), from the option each unit's answer
        picks (-1: refused). A refused unit is on no option and has no level."""

    def fake(self, q: Question, u: Uniform, machine: Answer | None) -> Answer:
        """The fake decider's answer: the machine's with noise, or noise alone without a machine form."""

    def seed(self, q: Question) -> dict[str, Any]:
        """What the fake decider's noise is seeded with: the question as version 0.1.0 hashed it,
        so that the fake answers what it answered then."""


class Predicate:
    token, key, takes_rules = "p", None, True

    def options(self, raw, where):
        return ("no", "yes"), (None, None)

    def wire(self, q):
        return {}

    def cell(self, a, q):
        return {"p": a.p}

    def answer(self, cell, q):
        return Yes(float(cell["p"]))

    def parse(self, item, q):
        if not _probability(item.get("probability")):
            raise ValueError(f"the probability {item.get('probability')!r} is not one")
        return Yes(float(item["probability"]))

    def v01(self, value, q):
        return Yes(float(value)) if _probability(value) else None

    def row(self, a, n):
        return [a.p]

    def features(self, rest):
        return log_ratios(rest)

    def soft(self, n, rng, size):
        return rng.random(size)[:, None]

    def sharp(self, n, rng, size):
        share = rng.uniform(*SHARP_YES_SHARE)
        return (rng.random(size) < share).astype(float)[:, None]

    def machine(self, q, value):
        return Yes(float(bool(value)))

    def picked(self, rest, threshold):
        """Yes at the threshold or above."""
        return (rest[:, 0] >= threshold).astype(int)

    def chosen(self, a):
        return -1 if isinstance(a, Refused) else int(a.p >= 0.5)

    def factors(self, q, picked):
        return [("yes", (picked == 1).astype(float))]

    def fake(self, q, u, machine):
        if machine is None:
            return Yes(round(u(), 2))
        hit = (machine.p == 1.0) != (u("flip") < 0.01)
        return Yes(round((0.94 if hit else 0.06) + (u("jit") - 0.5) * 0.08, 2))

    def seed(self, q):
        return {"type": "boolean", "instructions": q.instructions}


class _Listed:
    """What a choice and a score share: a list of options in the prompt file and on the wire,
    under `key`, each named by `item`, at most `most` of them."""
    takes_rules = False
    key: str
    item: str
    most: int

    def options(self, raw, where):
        items = raw.get(self.key)
        if not isinstance(items, list) or not 2 <= len(items) <= self.most:
            raise PlabError(f"{where}.{self.key} must be a list of 2 to {self.most}")
        items = [{self.item: x} if isinstance(x, str) else x for x in items]
        for k, x in enumerate(items):
            if (not isinstance(x, dict) or not isinstance(x.get(self.item), str) or not x[self.item]
                    or not set(x) <= {self.item, "description"} or not isinstance(x.get("description", ""), str)):
                raise PlabError(f'{where}.{self.key}[{k}] must be a string or {{"{self.item}": "...", "description": "..."}}')
        options = tuple(x[self.item] for x in items)
        if len(set(options)) != len(options):
            raise PlabError(f"{where}.{self.key} names a {self.item} twice")
        return options, tuple(x.get("description") for x in items)

    def wire(self, q):
        return {self.key: [{self.item: o, **({"description": d} if d is not None else {})}
                           for o, d in zip(q.options, q.descriptions)]}

    def picked(self, rest, threshold):
        """The option with the most probability (the first of a tie)."""
        return np.column_stack([1 - rest.sum(axis=1), rest]).argmax(axis=1)

    def chosen(self, a):
        return -1 if isinstance(a, Refused) else int(round(a.point))

    def _complete(self, given: Mapping[Any, Any], keys: Sequence[Any]) -> tuple[float, ...] | None:
        """The probability of each option, found by its key; None unless every option has one."""
        if set(given) != set(keys) or not all(_probability(p) for p in given.values()):
            return None
        return tuple(float(given[k]) for k in keys)

    def _given(self, item: Mapping[str, Any], keys: Sequence[Any]) -> tuple[float, ...] | None:
        """The API lists {"value", "probability"} per option, or nothing when the model gives none."""
        listed = item.get("probabilities")
        given = {x.get("value"): x.get("probability") for x in listed if isinstance(x, dict)} if isinstance(listed, list) else {}
        return self._complete(given, keys)

    def soft(self, n, rng, size):
        return rng.dirichlet(np.ones(n), size)[:, 1:]

    def sharp(self, n, rng, size):
        shares = rng.dirichlet(np.ones(n))
        return np.eye(n)[rng.choice(n, size, p=shares)][:, 1:]

    def _fake_probs(self, q, u, lean):
        raw = [u(o) for o in q.options]
        total = sum(raw) or 1.0
        if lean is None:
            return tuple(round(x / total, 2) for x in raw)
        if u("flip") < 0.01:
            lean = (lean + 1) % len(raw)
        return tuple(round(FAKE_LEAN * (j == lean) + (1 - FAKE_LEAN) * x / total, 2) for j, x in enumerate(raw))


class Choice(_Listed):
    token, key, item, most = "c", "choices", "value", 255

    def cell(self, a, q):
        return {"choice": q.options[a.point], **({"probs": list(a.probs)} if a.probs else {})}

    def answer(self, cell, q):
        return Chose(tuple(cell["probs"]) if "probs" in cell else None, q.options.index(cell["choice"]))

    def parse(self, item, q):
        if item.get("choice") not in q.options:
            raise ValueError(f"the choice {item.get('choice')!r} is not an option")
        return Chose(self._given(item, q.options), q.options.index(item["choice"]))

    def v01(self, value, q):
        """0.1 kept the probabilities by option, and not which option the model chose."""
        probs = self._complete(value, q.options) if isinstance(value, dict) else None
        return Chose(probs, probs.index(max(probs))) if probs else None

    def row(self, a, n):
        return list(a.probs[1:]) if a.probs else [float(j == a.point) for j in range(1, n)]

    def features(self, rest):
        return log_ratios(rest)

    def machine(self, q, value):
        return Chose(None, q.options.index(value)) if value in q.options else Refused()

    def factors(self, q, picked):
        """One indicator per option; of two options the second alone, the first being its complement."""
        return [(q.options[j], (picked == j).astype(float)) for j in (range(1, 2) if len(q.options) == 2 else range(len(q.options)))]

    def fake(self, q, u, machine):
        probs = self._fake_probs(q, u, machine.point if isinstance(machine, Chose) else None)
        return Chose(probs, max(range(len(probs)), key=probs.__getitem__))

    def seed(self, q):
        return {"type": "choice", "instructions": q.instructions, "criteria": dict(zip(q.options, q.descriptions))}


class Score(_Listed):
    token, key, item, most = "s", "levels", "label", 10

    def cell(self, a, q):
        return {"score": a.point, **({"probs": list(a.probs)} if a.probs else {})}

    def answer(self, cell, q):
        return Scored(tuple(cell["probs"]) if "probs" in cell else None, float(cell["score"]))

    def parse(self, item, q):
        """The API keys a level by its index, so the order of the list does not matter."""
        score, top = item.get("score"), len(q.options) - 1
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= top:
            raise ValueError(f"the score {score!r} is not a mean level index of 0 to {top}")
        return Scored(self._given(item, range(len(q.options))), float(score))

    def v01(self, value, q):
        raise PlabError(f"question {q.name} is a score, and 0.1 kept a score's answer only as score/(levels - 1), "
                        "without the probabilities of its levels")

    def row(self, a, n):
        """Without level probabilities, the two levels around the mean share it so that the mean holds."""
        if a.probs:
            return list(a.probs[1:])
        low = min(int(a.point), n - 2)
        up = a.point - low
        return [(1 - up) * (j == low) + up * (j == low + 1) for j in range(1, n)]

    def features(self, rest):
        """One column: the mean level over the top level, read like a predicate's probability."""
        top = rest.shape[1]
        return log_ratios((rest @ np.arange(1, top + 1) / top)[:, None])

    def machine(self, q, value):
        level = isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(q.options)
        return Scored(None, float(value)) if level else Refused()

    def factors(self, q, picked):
        """One ordered column: the level."""
        return [("level", np.where(picked >= 0, picked, np.nan).astype(float))]

    def fake(self, q, u, machine):
        probs = self._fake_probs(q, u, round(machine.point) if isinstance(machine, Scored) else None)
        return Scored(probs, float(sum(j * p for j, p in enumerate(probs))))

    def seed(self, q):
        return {"type": "score", "instructions": q.instructions, "criteria": list(q.options)}


KINDS: Mapping[str, Kind] = {"predicate": Predicate(), "choice": Choice(), "score": Score()}


def wire(q: Question) -> dict[str, Any]:
    """The question as the Decisions API takes it."""
    return {"type": q.kind, "name": q.name, "instructions": q.instructions, **KINDS[q.kind].wire(q)}


def to_cell(q: Question, a: Answer) -> dict[str, Any]:
    return {"refused": True} if isinstance(a, Refused) else KINDS[q.kind].cell(a, q)


def from_cell(q: Question, cell: Mapping[str, Any]) -> Answer:
    return Refused() if cell.get("refused") else KINDS[q.kind].answer(cell, q)


def parse_answer(q: Question, item: Mapping[str, Any]) -> Answer:
    """One entry of the "answers" of a Decisions API reply, as the answer to q."""
    return Refused() if item.get("type") == "refusal" else KINDS[q.kind].parse(item, q)


def _probability(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1


def machine_answer(q: Question, data: Mapping[str, Any]) -> Answer:
    return KINDS[q.kind].machine(q, q.machine_value(data))


def fake_answer(q: Question, u: Uniform, data: Mapping[str, Any]) -> Answer:
    return KINDS[q.kind].fake(q, u, machine_answer(q, data) if q.machine_ready else None)


@dataclass(frozen=True)
class Table:
    """The answers of N units to the questions of a prompt, or random answers of the same shape:
    per question an (N, options - 1) array in reference form."""
    spaces: tuple[Space, ...]
    rest: tuple[np.ndarray, ...]
    refused: tuple[np.ndarray, ...] = ()
    """Per question, where the decider declined; none when not given."""

    def __post_init__(self) -> None:
        if not self.refused:
            object.__setattr__(self, "refused", tuple(np.zeros(len(r), bool) for r in self.rest))

    @classmethod
    def of(cls, questions: Sequence[Question], answers: Sequence[Sequence[Answer]]) -> Table:
        """answers[i][j] is unit i's answer to question j. A refused answer is the uniform one."""
        rest, refused = [], []
        for j, q in enumerate(questions):
            n, kind = len(q.options), KINDS[q.kind]
            rows = [[1 / n] * (n - 1) if isinstance(a[j], Refused) else kind.row(a[j], n) for a in answers]
            rest.append(np.array(rows, float).reshape(len(answers), n - 1))
            refused.append(np.array([isinstance(a[j], Refused) for a in answers], bool))
        return cls(tuple(q.space for q in questions), tuple(rest), tuple(refused))

    @classmethod
    def random(cls, spaces: tuple[Space, ...], rng: np.random.Generator, size: int, *, sharp: bool) -> Table:
        """Random answers of a shape, question by question: spread out, or each on one option."""
        kinds = [KINDS[s.kind] for s in spaces]
        return cls(spaces, tuple((k.sharp if sharp else k.soft)(s.n, rng, size) for k, s in zip(kinds, spaces)))

    def features(self) -> np.ndarray:
        """What V reads, (N, d): the columns of every question side by side."""
        return np.column_stack([KINDS[s.kind].features(r) for s, r in zip(self.spaces, self.rest)])

    def sharp(self) -> np.ndarray:
        """Per unit, whether every answer has at most SHARP_WITHIN of its probability off one option."""
        off_top = [np.column_stack([r.sum(axis=1), 1 - r]).min(axis=1) for r in self.rest]
        return np.all([off <= SHARP_WITHIN for off in off_top], axis=0)
