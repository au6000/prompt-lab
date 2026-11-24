"""Prompt files: which fields of the state a decider reads, the questions it is asked about them,
and what its answer forecasts.

    {"name": "...", "fields": "all" | "<group>" | ["field", ...],
     "questions": [{"name", "type": "predicate", "rules": [{"text", "expr"}, "a rule in words", ...]},
                   {"name", "type": "choice", "instructions", "choices": [...], "expr"}, ...],
     "forecast": {"question": "<name>", "classes": {"<option>": "<outcome class>"}} | "fitted"}

The two forms of version 0.1 (veto_rules; questions as an object beside "score") are rewritten into
this form before anything is parsed, so nothing past this file knows them.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .expr import Expr, ExprError
from .layout import PlabError, short_hash
from .questions import KINDS, Question, Rule, wire
from .rows import Row
from .task import TaskConfig

MAX_QUESTIONS = 6
MAX_RULES = 12
MAX_INSTRUCTIONS = 4000
V01_TYPES = {"boolean": "predicate", "choice": "choice", "score": "score"}
V01_POSITIVE = {"positive": True, "loss": True, "negative": False, "win": False}


@dataclass(frozen=True)
class QuestionForecast:
    """The prompt's statement of what an answer says about the outcome: option j of one question
    forecasts outcome class option_class[j]."""
    question: int
    option_class: tuple[int, ...]


Forecast = QuestionForecast | Literal["fitted"] | None
"""None: the prompt is measured by the information in its answers only. "fitted": V fitted on the
training answers is the forecast, so there is no as-is gain."""


@dataclass(frozen=True)
class Ask:
    """One decider call: the values one unit gives, in the prompt's field order, and the questions."""
    unit: str
    data: Mapping[str, Any]
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class ReaderPrompt:
    name: str
    file: str | None
    raw: Mapping[str, Any]
    fields: tuple[str, ...]
    questions: tuple[Question, ...]
    forecast: Forecast

    @property
    def prompt_id(self) -> str:
        """A hash of everything that can change a measurement: the fields, the questions as asked,
        their machine forms and the forecast. Not the name."""
        fc = self.forecast
        return short_hash({
            "fields": self.fields,
            "questions": [{**wire(q), "exprs": [r.expr and r.expr.src for r in q.rules], "expr": q.expr and q.expr.src}
                          for q in self.questions],
            "forecast": [fc.question, fc.option_class] if isinstance(fc, QuestionForecast) else fc})

    @property
    def rules(self) -> tuple[Rule, ...]:
        return tuple(r for q in self.questions for r in q.rules)

    def asks(self, rows: Sequence[Row]) -> list[Ask]:
        return [Ask(r.id, {f: r.state.get(f) for f in self.fields}, self.questions) for r in rows]

    def rule_hits(self, rows: Sequence[Row]) -> list[np.ndarray | None]:
        """Per rule, where its machine form applies; None for a rule without one."""
        data = [a.data for a in self.asks(rows)]
        return [np.array([r.expr.applies(d) for d in data], bool) if r.expr else None for r in self.rules]

    def per_rule(self, frame: tuple[str, str]) -> ReaderPrompt:
        """The same fields, asked one rule at a time: a question rule_<number> per rule."""
        return replace(self, forecast=None, questions=tuple(
            Question(f"rule_{r.number}", q.kind, frame[0] + r.text + frame[1], q.options, q.descriptions, (r,), None)
            for q in self.questions for r in q.rules))


def _v01_question(name: str, q: Any) -> dict[str, Any]:
    """One entry of the questions object of version 0.1, as a question in today's form."""
    q = dict(q) if isinstance(q, dict) else {}
    criteria = q.pop("criteria", None)
    if q.get("type") == "choice" and isinstance(criteria, dict):
        q["choices"] = [{"value": k, "description": v} for k, v in criteria.items()]
    elif q.get("type") == "score":
        q["levels"] = criteria
    return {**q, "name": name, "type": V01_TYPES.get(q.get("type"))}


def v01_questions(sent: Mapping[str, Any]) -> tuple[Question, ...]:
    """The questions of a version 0.1 answer cache. Its meta file keeps them as they were sent, with
    the rule frame already around the rules."""
    return tuple(_question(_v01_question(name, q), f"questions.{name}", (), ("", ""), 0) for name, q in sent.items())


def _from_v01(raw: Mapping[str, Any], classes: tuple[str, ...]) -> dict[str, Any]:
    """A prompt file of version 0.1 as the same prompt in today's form."""
    if len(classes) != 2:
        raise PlabError("veto_rules and questions with \"score\" are the two-class forms of version 0.1; "
                        "with more outcome classes write questions as a list, and a forecast")
    rest = {k: v for k, v in raw.items() if k not in ("form", "veto_rules", "questions", "score")}
    if "veto_rules" in raw:
        if raw.get("form", "veto") != "veto" or "questions" in raw:
            raise PlabError("a prompt has either veto_rules or questions, not both")
        return {**rest, "questions": [{"name": "skip_veto", "type": "predicate", "rules": raw["veto_rules"]}],
                "forecast": {"question": "skip_veto", "classes": {"yes": classes[1], "no": classes[0]}}}
    old, score = raw["questions"], raw.get("score")
    questions = [_v01_question(name, q) for name, q in old.items()]
    if score == {"combine": "lr"}:
        return {**rest, "questions": questions, "forecast": "fitted"}
    target = old.get(score.get("q")) if isinstance(score, dict) else None
    if not isinstance(target, dict) or score.get("high_means") not in V01_POSITIVE:
        raise PlabError('a prompt with questions as an object needs score: {"q": <a question>, "high_means": "positive" | "negative"} '
                        'or {"combine": "lr"}')
    yes, no = (classes[1], classes[0]) if V01_POSITIVE[score["high_means"]] else (classes[0], classes[1])
    if target.get("type") == "boolean":
        mapping = {"yes": yes, "no": no}
    elif target.get("type") == "choice" and isinstance(target.get("criteria"), dict) and score.get("option") in target["criteria"]:
        mapping = {o: yes if o == score["option"] else no for o in target["criteria"]}
    else:
        raise PlabError("score.q must name a boolean question, or a choice question with score.option one of its options. "
                        "For a score question write a forecast that says which levels forecast which class")
    return {**rest, "questions": questions, "forecast": {"question": score["q"], "classes": mapping}}


def _fields(spec: Any, task: TaskConfig, all_fields: tuple[str, ...]) -> tuple[str, ...]:
    if spec == "all":
        return all_fields
    if isinstance(spec, str):
        if spec not in task.field_groups:
            raise PlabError(f"fields: unknown group {spec!r} (task.json field_groups has {sorted(task.field_groups)})")
        spec = list(task.field_groups[spec])
    if not isinstance(spec, list) or not spec or not all(isinstance(x, str) for x in spec):
        raise PlabError("fields must be \"all\", a group name, or a list of field names")
    missing = [k for k in spec if k not in all_fields]
    if missing:
        raise PlabError(f"fields not in the task's state fields: {missing} (see `plab fields`)")
    if len(set(spec)) != len(spec):
        raise PlabError("fields lists a name twice")
    return tuple(spec)


def _expr(src: Any, where: str, fields: tuple[str, ...]) -> Expr | None:
    if src is None:
        return None
    try:
        expr = Expr(src)
    except ExprError as e:
        raise PlabError(f"{where}: {e}")
    extra = sorted(expr.names - set(fields))
    if extra:
        raise PlabError(f"{where} reads fields not given to the decider: {extra} "
                        "(add them to fields, or the machine would see more than the model)")
    return expr


def _rules(raw: Any, where: str, fields: tuple[str, ...], before: int) -> tuple[Rule, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_RULES:
        raise PlabError(f"{where}.rules must be a list of 1 to {MAX_RULES} rules")
    rules = []
    for number, r in enumerate(raw, before + 1):
        r = {"text": r} if isinstance(r, str) else r
        if not isinstance(r, dict) or not set(r) <= {"text", "expr"} or not isinstance(r.get("text"), str) or not r["text"].strip():
            raise PlabError(f"rule {number} must be a string or {{\"text\": \"...\", \"expr\": \"...\"}} with a text")
        rules.append(Rule(number, r["text"].strip(), _expr(r.get("expr"), f"rule {number}: expr", fields)))
    return tuple(rules)


def _question(raw: Any, where: str, fields: tuple[str, ...], frame: tuple[str, str], before: int) -> Question:
    if not isinstance(raw, dict) or raw.get("type") not in KINDS:
        raise PlabError(f"{where}.type must be one of {', '.join(KINDS)}")
    kind = KINDS[raw["type"]]
    unknown = sorted(set(raw) - {"name", "type", "instructions", "rules", "expr", kind.key})
    if unknown:
        raise PlabError(f"{where} does not take {unknown}")
    if not isinstance(raw.get("name"), str) or not raw["name"]:
        raise PlabError(f"{where}.name must be a non-empty string")
    options, descriptions = kind.options(raw, where)
    if "rules" in raw:
        if not kind.takes_rules or "instructions" in raw or "expr" in raw:
            raise PlabError(f"{where}: rules belong to a predicate, in place of its instructions and expr")
        rules, expr = _rules(raw["rules"], where, fields, before), None
        instructions = frame[0] + " ".join(r.text for r in rules) + frame[1]
    else:
        rules, instructions, expr = (), raw.get("instructions"), _expr(raw.get("expr"), f"{where}.expr", fields)
        if not isinstance(instructions, str):
            raise PlabError(f"{where}.instructions must be a string")
    if len(instructions) > MAX_INSTRUCTIONS:
        raise PlabError(f"{where}: instructions longer than {MAX_INSTRUCTIONS} characters")
    return Question(raw["name"], raw["type"], instructions, options, descriptions, rules, expr)


def _forecast(raw: Any, questions: Sequence[Question], classes: tuple[str, ...]) -> Forecast:
    if raw is None or raw == "fitted":
        return raw
    names = [q.name for q in questions]
    if not isinstance(raw, dict) or set(raw) != {"question", "classes"} or raw["question"] not in names or not isinstance(raw["classes"], dict):
        raise PlabError('forecast must be "fitted" or {"question": <a question name>, "classes": {<option>: <outcome class>}}')
    q = questions[names.index(raw["question"])]
    given = raw["classes"]
    if set(given) != set(q.options) or not all(c in classes for c in given.values()):
        raise PlabError(f"forecast.classes must give one of the outcome classes ({', '.join(classes)}) "
                        f"for every option of {q.name} ({', '.join(q.options)})")
    return QuestionForecast(names.index(q.name), tuple(classes.index(given[o]) for o in q.options))


def parse_prompt(raw: Any, task: TaskConfig, all_fields: tuple[str, ...], *, name: str, file: str | None = None) -> ReaderPrompt:
    """A prompt from the JSON of its file. `name` is used when the file gives none."""
    if not isinstance(raw, dict):
        raise PlabError("a prompt file must hold a JSON object")
    spec = {k: v for k, v in raw.items() if not k.startswith("_")}
    if "veto_rules" in spec or isinstance(spec.get("questions"), dict):
        spec = _from_v01(spec, task.outcome.classes)
    unknown = sorted(set(spec) - {"name", "fields", "questions", "forecast"})
    if unknown:
        raise PlabError(f"a prompt has name, fields, questions and forecast; it does not take {unknown} (a key starting with _ is a note)")
    fields = _fields(spec.get("fields", "all"), task, all_fields)
    listed = spec.get("questions")
    if not isinstance(listed, list) or not 1 <= len(listed) <= MAX_QUESTIONS:
        raise PlabError(f"questions must be a list of 1 to {MAX_QUESTIONS} questions")
    questions: list[Question] = []
    for k, q in enumerate(listed):
        questions.append(_question(q, f"questions[{k}]", fields, task.rule_frame, sum(len(x.rules) for x in questions)))
    if len({q.name for q in questions}) != len(questions):
        raise PlabError("two questions share a name")
    return ReaderPrompt(str(spec.get("name") or name), file, raw, fields, tuple(questions),
                        _forecast(spec.get("forecast"), questions, task.outcome.classes))


def load_prompt(path: str, task: TaskConfig, all_fields: tuple[str, ...]) -> ReaderPrompt:
    """Read a prompt file. A path that is not found from the current folder is tried from the task folder."""
    p = Path(path)
    if not p.exists() and not p.is_absolute() and (task.root / p).exists():
        p = task.root / p
    if not p.exists():
        raise PlabError(f"prompt file not found: {path}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PlabError(f"{path}: not valid JSON: {e}")
    p = p.resolve()
    file = p.relative_to(task.root).as_posix() if p.is_relative_to(task.root) else str(p)
    return parse_prompt(raw, task, all_fields, name=p.stem, file=file)


_AND_RE = re.compile(r"\b(and|both|all of)\b", re.IGNORECASE)
_TIMES_RE = re.compile(r"(?<![\d.])0?\.\d+\s*(x|times|\*)(?![\w])", re.IGNORECASE)
_LISTMAX_RE = re.compile(r"\b(max(imum)?|min(imum)?|highest|lowest|largest|smallest|any|all)\b[^.]{0,60}\b(of|among|in)\b[^.]{0,40}"
                         r"\b(last|list|series|closes|values)\b", re.IGNORECASE)


def _multiplies_below_one(expr: Expr) -> bool:
    return any(isinstance(side, ast.Constant) and isinstance(side.value, (int, float)) and not isinstance(side.value, bool)
               and 0 < abs(side.value) < 1
               for node in ast.walk(expr.tree) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
               for side in (node.left, node.right))


def _list_reductions(expr: Expr, list_len: Mapping[str, int]) -> list[tuple[str, str, int | None]]:
    """(function, field, length) for each max or min the expr takes over a list field or a slice of one."""
    out = []
    for node in ast.walk(expr.tree):
        if isinstance(node, ast.Call) and node.func.id in ("max", "min") and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id in list_len:
                out.append((node.func.id, arg.id, list_len[arg.id]))
            if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name) and arg.value.id in list_len:
                out.append((node.func.id, arg.value.id, None))
    return out


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _compares_with_none(expr: Expr) -> bool:
    """Whether the expr tests a value against None (`x == None`, `x != None`, `x in (None, ...)`).
    A None field leaves the expr without a value, so such a test never sees one. `None in LIST`
    is left alone: an element of a list that has a value may be None."""
    for node in ast.walk(expr.tree):
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            for left, op, right in zip(sides, node.ops, sides[1:]):
                if isinstance(op, (ast.Eq, ast.NotEq)) and (_is_none(left) or _is_none(right)):
                    return True
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List)) and any(map(_is_none, right.elts)):
                    return True
    return False


def lint(prompt: ReaderPrompt, train: Sequence[Row], all_fields: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """(warnings, notes) about wordings that made the decision model follow a rule less like the
    machine in the case study (docs/protocol.md, lesson 3), and about what the prompt leaves out."""
    warns, notes = [], []
    list_len: dict[str, int] = {}
    for r in train[:500]:
        for k in prompt.fields:
            if isinstance(r.state.get(k), list):
                list_len[k] = max(list_len.get(k, 0), len(r.state[k]))
    if prompt.forecast is None:
        notes.append("no forecast: the prompt is measured by the information in its answers alone: cv_gain, and transfer_gain "
                     "off the training split (no as-is gain, AUC or decision table)")
    for q in prompt.questions:
        for tag, e in ((f"question {q.name}", q.expr), *((f"rule {r.number}", r.expr) for r in q.rules)):
            if e is not None and _compares_with_none(e):
                warns.append(f"{tag}: the expr compares with None. When a field it reads is None the expr has no value "
                             "(a rule does not apply, a yes/no question answers no, a choice or a score is refused), "
                             "so `== None` and `in (None, ...)` never match a None, and `!= None` holds whenever the field has a value")
        if len(q.instructions) > 3000:
            warns.append(f"question {q.name}: instructions are {len(q.instructions)} characters (limit {MAX_INSTRUCTIONS})")
        if not q.rules and q.expr is None:
            notes.append(f"question {q.name}: no expr, so the machine decider cannot answer it")
    n_and = 0
    for r in prompt.rules:
        tag = f"rule {r.number}"
        n_and += bool(_AND_RE.search(r.text)) or (r.expr is not None and " and " in f" {r.expr.src} ")
        if _TIMES_RE.search(r.text) or (r.expr is not None and _multiplies_below_one(r.expr)):
            warns.append(f"{tag}: multiplies by a number below 1. In the case study the model followed such rules "
                         "only 59-95% like the machine; prefer a plain threshold or a ratio stated in words checked with `plab agree`")
        reductions = _list_reductions(r.expr, list_len) if r.expr is not None else []
        for fn, field, length in reductions:
            if length is None or length >= 12:
                warns.append(f"{tag}: takes {fn}() over the list field {field}{f' ({length} values)' if length else ''}. "
                             "Reductions over long lists (12 values in the case study) lowered agreement")
        if not reductions and _LISTMAX_RE.search(r.text) and any(f in r.text for f in list_len):
            warns.append(f"{tag}: the text asks for a maximum/any over a list field; reductions over long lists lowered agreement")
        if r.expr is None:
            notes.append(f"{tag}: no expr, so the machine decider cannot answer it (and `plab agree` cannot check it)")
        mentioned = sorted(f for f in all_fields if f in r.text and f not in prompt.fields and len(f) > 3)
        if mentioned:
            warns.append(f"{tag}: the text names fields that are not given to the decider: {mentioned}")
        unnamed = [f for f in (r.expr.names if r.expr else ()) if f not in r.text]
        if unnamed:
            notes.append(f"{tag}: expr reads {unnamed}, which the text does not name; make sure the words say the same thing")
    if n_and >= 4:
        warns.append(f"{n_and} rules contain 'and'/'both'. With 4 or more such rules the model agreed with the machine "
                     "only 59-95% in the case study; split or simplify, then check with `plab agree`")
    read = {f for q in prompt.questions for e in (q.expr, *(r.expr for r in q.rules)) if e for f in e.names}
    unused = [f for f in prompt.fields if f not in read and not any(f in q.instructions for q in prompt.questions)]
    if unused and len(unused) < len(prompt.fields):
        notes.append(f"fields given but not used by any question: {unused} (they still go to the decider)")
    return warns, notes
