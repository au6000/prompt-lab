"""The task: task.json, read once into typed settings.

DEFAULTS is the only place a default lives; the dataclasses below have none. A task.json is merged
over it (objects merge, lists and the REPLACED keys replace). The pre-registration lock covers the
merged settings except UNLOCKED, so a setting added later is locked unless it is listed there.
"""
from __future__ import annotations

import copy
import json
import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from .layout import PlabError

FORMAT = 2
SPLITS = ("train", "eval", "test")

DEFAULTS: dict[str, Any] = {
    "format": FORMAT,
    "name": "task",
    "rows": {"file": "rows.jsonl", "id": "id", "time": "t", "outcome": "label", "value": None, "state": "state"},
    "outcome": {"classes": ["0", "1"], "meaning": {}},
    "splits": None,
    "cv_groups": "month",
    "half_boundary": None,
    "train_sample": None,
    "decision": None,
    "state_fields": "all",
    "field_groups": {},
    "rule_frame": {
        "head": ("These are veto rules (reasons not to act). If at least one of them applies to this case, the case is "
                 "skipped; if none of them applies, the case is taken. Apply the rules to the values given. "),
        "tail": " Following these rules, should this case be skipped?",
    },
    "selection": {
        "require": [{"metric": "train.cv_gain_pct", "op": ">", "value": 0},
                    {"metric": "eval.cv_gain_pct", "op": ">", "value": 0},
                    {"metric": "eval.decision.value_per_1000", "op": ">", "value": 0}],
        "rank_by": [{"metric": "eval.decision.value_per_1000", "order": "desc"},
                    {"metric": "eval.decision.skip_frac", "order": "asc"}],
    },
    "budget": {"usd": 0.0, "eval_checks_per_agent": 10, "nominations_per_agent": 2},
    "deciders": {"machine": {"kind": "machine"},
                 "fake": {"kind": "fake", "cost_per_call": 0.0},
                 "jev": {"kind": "gateway", "model": "typesafe-ai/jev", "price_per_mtok": 0.042}},
    "decider": "jev",
    "gateway": {"workers": 6, "retries": 4, "timeout_s": 120},
    "metrics": {"cv_folds": 10, "cv_reps": 3, "mi_perms": 500, "null_reps": 200, "null_seed": 777},
}

#: Objects that a task.json value replaces whole instead of merging into the default.
REPLACED = ("selection",)
#: Settings that cannot change a measurement. "*" stands for any decider name.
UNLOCKED = ("name", "budget.usd", "gateway", "deciders.*.price_per_mtok", "deciders.*.cost_per_call")
#: The comparisons a requirement of the selection rule may make.
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne}

#: The numbers a selection rule may read from a measurement, after "train." or "eval.", and what
#: the task must have for a measurement to hold them. A prompt can still lack some (without a
#: forecast, all but the gains; a fitted forecast on the training split, mi_pct, mi_p and xfit:
#: ADR-0012), and a missing number fails its requirement.
TRAIN_ONLY = "the training split"
NOT_TRAIN = "a split other than training"
_DECISION = ("skip_frac", "kept_pos_pct", "kept_neg_pct", "skipped_pos_pct", "lift", "value_per_1000",
             "all_in_value_per_1000", "gain_vs_all_in_per_1000")
SELECTABLE: Mapping[str, tuple[str, ...]] = {
    **dict.fromkeys(("cv_gain_pct", "asis_gain_pct", "null_pct.cv_gain_pct", "n", "missing", "refused"), ()),
    **dict.fromkeys(("transfer_gain_pct", "null_pct.transfer_gain_pct"), (NOT_TRAIN,)),
    **dict.fromkeys(("auc", "mi_pct", "mi_p", "null_pct.auc", "null_pct.mi_pct"), ("two classes",)),
    **dict.fromkeys(("auc_h1", "auc_h2"), ("two classes", "half_boundary", TRAIN_ONLY)),
    **{f"decision.{k}": ("decision",) for k in _DECISION},
    **{f"xfit.{k}": ("two classes", "rows.value", "half_boundary", TRAIN_ONLY) for k in ("value_per_1000", "keep_frac", "kept_neg_pct")},
}


@dataclass(frozen=True)
class RowsSpec:
    """Where each part of a row lives in the rows file. value is None when rows carry no value."""
    file: str
    id: str
    time: str
    outcome: str
    value: str | None
    state: str


@dataclass(frozen=True)
class OutcomeSpec:
    """The K >= 2 outcome classes in index order. Class 0 is the reference class: the common case,
    the one the training down-sample thins. With two classes, class 1 is the positive class."""
    classes: tuple[str, ...]
    meaning: Mapping[str, str]

    @property
    def K(self) -> int:
        return len(self.classes)

    def index(self, raw: Any) -> int:
        """The class of an outcome as a row writes it: true and false read as 1 and 0, a whole
        number as its integer, then matched by its text."""
        if isinstance(raw, bool) or (isinstance(raw, float) and raw.is_integer()):
            raw = int(raw)
        if isinstance(raw, (int, str)) and str(raw) in self.classes:
            return self.classes.index(str(raw))
        raise PlabError(f"outcome {raw!r} is not one of outcome.classes {list(self.classes)}")


@dataclass(frozen=True)
class SampleSpec:
    """The training down-sample: every unit outside the reference class, and reference_per_other
    reference units for each of them, drawn with default_rng(seed). Weights restore the population."""
    reference_per_other: float
    seed: int


@dataclass(frozen=True)
class DecisionSpec:
    """The veto use: a unit is skipped when its forecast of the positive class reaches this."""
    skip_if_prob_at_least: float


@dataclass(frozen=True)
class Budget:
    usd: float
    eval_checks_per_agent: int
    nominations_per_agent: int


@dataclass(frozen=True)
class DeciderSpec:
    """One entry of `deciders`: its name, its kind, and the settings DECIDER_SETTINGS lists for the kind."""
    name: str
    kind: str
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class GatewaySettings:
    """How the AI Gateway is called: parallel calls, retries of a failed call, seconds per call."""
    workers: int
    retries: int
    timeout_s: float


@dataclass(frozen=True)
class MetricSettings:
    cv_folds: int
    cv_reps: int
    mi_perms: int
    null_reps: int
    null_seed: int


@dataclass(frozen=True)
class TaskConfig:
    root: Path
    merged: Mapping[str, Any]
    name: str
    rows: RowsSpec
    outcome: OutcomeSpec
    splits: Mapping[str, tuple[datetime, datetime]]
    cv_groups: str
    half_boundary: datetime | None
    train_sample: SampleSpec | None
    decision: DecisionSpec | None
    state_fields: Literal["all"] | tuple[str, ...]
    field_groups: Mapping[str, tuple[str, ...]]
    rule_frame: tuple[str, str]
    selection: Mapping[str, Any]
    budget: Budget
    deciders: Mapping[str, DeciderSpec]
    decider: str
    gateway: GatewaySettings
    metrics: MetricSettings

    def locked(self) -> dict[str, Any]:
        """The merged settings without the UNLOCKED ones: what prereg.lock hashes and stores."""
        out = copy.deepcopy(dict(self.merged))
        for path in UNLOCKED:
            _drop(out, path.split("."))
        return out


def _drop(obj: dict[str, Any], path: list[str]) -> None:
    head, rest = path[0], path[1:]
    for key in list(obj) if head == "*" else [head]:
        if key in obj and not rest:
            del obj[key]
        elif isinstance(obj.get(key), dict):
            _drop(obj[key], rest)


def merge(base: Mapping[str, Any], over: Mapping[str, Any], path: str = "") -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and path + k not in REPLACED:
            out[k] = merge(out[k], v, f"{path}{k}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_time(s: Any) -> datetime:
    """An ISO date or datetime, or epoch seconds, as an aware UTC datetime. A time without a zone
    is read as UTC; a trailing Z is accepted."""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    if not isinstance(s, str) or not s:
        raise PlabError(f"not a time: {s!r}")
    x = s.strip()
    if x[-1:] in ("Z", "z"):
        x = x[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(x)
    except ValueError:
        raise PlabError(f"not an ISO time: {s!r}")
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _fail(path: str, must: str) -> PlabError:
    return PlabError(f"task.json: {path} must be {must}")


def _obj(x: Any, path: str, keys: Iterable[str]) -> dict[str, Any]:
    keys = tuple(keys)
    if not isinstance(x, dict) or set(x) != set(keys):
        raise _fail(path, f"an object with exactly the keys {', '.join(keys)}")
    return x


def _number(x: Any, path: str, *, least: float | None = None, above: float | None = None) -> float:
    if isinstance(x, (int, float)) and not isinstance(x, bool) and (least is None or x >= least) and (above is None or x > above):
        return x
    bound = f" of at least {least}" if least is not None else f" above {above}" if above is not None else ""
    raise _fail(path, "a number" + bound)


def _count(x: Any, path: str, *, least: int = 0) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < least:
        raise _fail(path, f"an integer of at least {least}")
    return x


def _text(x: Any, path: str) -> str:
    if not isinstance(x, str) or not x:
        raise _fail(path, "a non-empty string")
    return x


def _names(x: Any, path: str) -> tuple[str, ...]:
    if not isinstance(x, list) or not x or not all(isinstance(v, str) and v for v in x) or len(set(x)) != len(x):
        raise _fail(path, "a list of distinct names")
    return tuple(x)


def _money(x: Any, path: str) -> float:
    return _number(x, path, least=0)


#: What each kind of decider takes in `deciders` beside "kind", and how each setting is checked.
#: Every setting is required: a decider that a task adds has no defaults.
DECIDER_SETTINGS: Mapping[str, Mapping[str, Callable[[Any, str], Any]]] = {
    "machine": {},
    "fake": {"cost_per_call": _money},
    "gateway": {"model": _text, "price_per_mtok": _money},
}


def _rows(x: Any) -> RowsSpec:
    rows = RowsSpec(**_obj(x, "rows", DEFAULTS["rows"]))
    for key in DEFAULTS["rows"]:
        if key != "value" or rows.value is not None:
            _text(getattr(rows, key), f"rows.{key}")
    return rows


def _outcome(x: Any) -> OutcomeSpec:
    _obj(x, "outcome", ("classes", "meaning"))
    classes = _names(x["classes"], "outcome.classes")
    if len(classes) < 2 or not isinstance(x["meaning"], dict) or not set(x["meaning"]) <= set(classes):
        raise _fail("outcome", "two or more classes, and a meaning for none but those classes")
    return OutcomeSpec(classes, x["meaning"])


def _splits(x: Any) -> dict[str, tuple[datetime, datetime]]:
    if not isinstance(x, dict) or not {"train", "eval"} <= set(x) <= set(SPLITS):
        raise _fail("splits", 'train and eval (and test if there is one), e.g. {"train": ["2018-01-01", "2022-01-01"], ...}')
    out: dict[str, tuple[datetime, datetime]] = {}
    for name in SPLITS:
        if x.get(name) is None:
            continue
        if not isinstance(x[name], list) or len(x[name]) != 2:
            raise _fail(f"splits.{name}", "[start, end)")
        start, end = parse_time(x[name][0]), parse_time(x[name][1])
        if not start < end:
            raise _fail(f"splits.{name}", "[start, end) with the start before the end")
        if out and start < list(out.values())[-1][1]:
            raise PlabError(f"task.json: splits.{name} overlaps the split before it (they must be in time order: train, eval, test)")
        out[name] = (start, end)
    return out


def _sample(x: Any) -> SampleSpec | None:
    if x is None:
        return None
    _obj(x, "train_sample", ("reference_per_other", "seed"))
    return SampleSpec(_number(x["reference_per_other"], "train_sample.reference_per_other", above=0),
                      _count(x["seed"], "train_sample.seed"))


def _decision(x: Any, outcome: OutcomeSpec, rows: RowsSpec) -> DecisionSpec | None:
    if x is None:
        return None
    if outcome.K != 2 or rows.value is None:
        raise PlabError("task.json: decision is the veto use; it needs two outcome classes and rows.value (or set decision to null)")
    _obj(x, "decision", ("skip_if_prob_at_least",))
    return DecisionSpec(float(_number(x["skip_if_prob_at_least"], "decision.skip_if_prob_at_least")))


def _deciders(x: Any) -> dict[str, DeciderSpec]:
    if not isinstance(x, dict):
        raise _fail("deciders", "an object of name: settings")
    out = {}
    for name, d in x.items():
        kind = d.get("kind") if isinstance(d, dict) else None
        if kind not in DECIDER_SETTINGS:
            raise _fail(f"deciders.{name}.kind", "one of " + ", ".join(DECIDER_SETTINGS))
        _obj(d, f"deciders.{name}", ("kind", *DECIDER_SETTINGS[kind]))
        settings = {key: check(d[key], f"deciders.{name}.{key}") for key, check in DECIDER_SETTINGS[kind].items()}
        out[name] = DeciderSpec(name, kind, settings)
    return out


def _selection(x: Any, task_has: Mapping[str, bool], classes: tuple[str, ...]) -> dict[str, Any]:
    """The selection rule, refused when it reads a number no measurement of this task can hold."""
    _obj(x, "selection", ("require", "rank_by"))
    for part, keys in (("require", ("metric", "op", "value")), ("rank_by", ("metric", "order"))):
        if not isinstance(x[part], list):
            raise _fail(f"selection.{part}", "a list")
        for k, item in enumerate(x[part]):
            where = f"selection.{part}[{k}]"
            metric = _text(_obj(item, where, keys)["metric"], f"{where}.metric")
            split, _, name = metric.partition(".")
            if split not in ("train", "eval") or not (name in SELECTABLE or name in [f"class_counts.{c}" for c in classes]):
                raise _fail(f"{where}.metric", "train. or eval. and then one of " + ", ".join([*SELECTABLE, "class_counts.<class>"])
                            + f" (it is {metric!r})")
            has = {**task_has, TRAIN_ONLY: split == "train", NOT_TRAIN: split != "train"}
            lacking = [need for need in SELECTABLE.get(name, ()) if not has[need]]
            if lacking:
                raise PlabError(f"task.json: {where} reads {metric}, which this task cannot produce (it needs {' and '.join(lacking)}). "
                                "State a selection rule the task can meet")
            if part == "rank_by" and item["order"] not in ("asc", "desc"):
                raise _fail(f"{where}.order", "asc or desc")
            if part == "require" and item["op"] not in OPS:
                raise _fail(f"{where}.op", "one of " + " ".join(OPS))
            if part == "require":
                _number(item["value"], f"{where}.value")
    return x


def parse_task(root: Path) -> TaskConfig:
    """Read and check <root>/task.json. The only place task.json is read."""
    root = Path(root).resolve()
    if not (root / "task.json").exists():
        raise PlabError(f"no task.json in {root} (make one with: plab init <dir>)")
    try:
        raw = json.loads((root / "task.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PlabError(f"task.json is not valid JSON: {e}")
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise PlabError(f'task.json has no "format": {FORMAT}: it was written for prompt-lab 0.1. Convert the folder with `plab migrate` '
                        "(it prints what it would change; add --write to do it)")
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise PlabError(f"task.json: unknown keys {unknown} (known: {', '.join(DEFAULTS)}; a key starting with _ is a note)")
    c = merge(DEFAULTS, raw)
    rows, outcome = _rows(c["rows"]), _outcome(c["outcome"])
    decision = _decision(c["decision"], outcome, rows)
    half = parse_time(c["half_boundary"]) if c["half_boundary"] is not None else None
    deciders = _deciders(c["deciders"])
    if c["decider"] not in deciders:
        raise _fail("decider", "one of the deciders: " + ", ".join(deciders))
    if not isinstance(c["field_groups"], dict):
        raise _fail("field_groups", "an object of name: [fields]")
    frame = _obj(c["rule_frame"], "rule_frame", ("head", "tail"))
    budget = _obj(c["budget"], "budget", DEFAULTS["budget"])
    gateway = _obj(c["gateway"], "gateway", DEFAULTS["gateway"])
    metrics = _obj(c["metrics"], "metrics", DEFAULTS["metrics"])
    task_has = {"two classes": outcome.K == 2, "rows.value": rows.value is not None, "decision": decision is not None,
                "half_boundary": half is not None}
    return TaskConfig(
        root=root, merged=c, name=_text(c["name"], "name"), rows=rows, outcome=outcome, splits=_splits(c["splits"]),
        cv_groups=_text(c["cv_groups"], "cv_groups"), half_boundary=half, train_sample=_sample(c["train_sample"]), decision=decision,
        state_fields="all" if c["state_fields"] == "all" else _names(c["state_fields"], "state_fields"),
        field_groups={k: _names(v, f"field_groups.{k}") for k, v in c["field_groups"].items()},
        rule_frame=(_text(frame["head"], "rule_frame.head"), _text(frame["tail"], "rule_frame.tail")),
        selection=_selection(c["selection"], task_has, outcome.classes),
        budget=Budget(float(_money(budget["usd"], "budget.usd")), _count(budget["eval_checks_per_agent"], "budget.eval_checks_per_agent"),
                      _count(budget["nominations_per_agent"], "budget.nominations_per_agent")),
        deciders=deciders, decider=c["decider"],
        gateway=GatewaySettings(_count(gateway["workers"], "gateway.workers", least=1), _count(gateway["retries"], "gateway.retries"),
                                float(_number(gateway["timeout_s"], "gateway.timeout_s", above=0))),
        metrics=MetricSettings(_count(metrics["cv_folds"], "metrics.cv_folds", least=2), _count(metrics["cv_reps"], "metrics.cv_reps", least=1),
                               _count(metrics["mi_perms"], "metrics.mi_perms", least=1),
                               _count(metrics["null_reps"], "metrics.null_reps", least=1), _count(metrics["null_seed"], "metrics.null_seed")))
