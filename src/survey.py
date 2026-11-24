"""The survey: does a task's data carry information that holds across time? (README.ja.md, section 7)

A factor is one column of numbers over the units: a state field, a level of a text field, a rule of
a prompt, an option of a question the machine can answer. For each factor the survey fits
logit P(class | F) = a + b x in every period, x being the factor's midrank, standardized, and reads

  mu, sigma    the mean of the true slope across periods and its between-period spread, with the
               sampling noise of each period taken out (DerSimonian-Laird random effects)
  share        mu2 / (mu2 + sigma^2), mu2 = max(0, mu^2 - se(mu)^2): of the information a slope known
               in every period could use, the share one fixed slope can use (weak-effect form)
  relearning   the gain when a + b x is re-learned every month on the last 6, 12, 24 or all earlier
               months, against re-learning the base rate alone; every window forecasts the same months

Every row is read at weight 1. The factors cost nothing (state values and exprs), so nothing is
down-sampled and no stored answer is read. With K > 2 outcome classes the same question is asked of
each non-reference class against the reference class, on the units of the two; with two classes that
is the positive class against the other, on every unit.

Which rows: the training split. Every split only when the study is closed, through the Lock of
protocol.enter(study, "survey"); the record then says "exploratory": true and is logged.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import __version__, protocol, yardsticks as Y
from .layout import AgentName, PlabError
from .prompt import ReaderPrompt
from .questions import KINDS, machine_answer
from .rows import Row, period_key
from .study import Study

# The survey has no setting in task.json: a key there would enter the pre-registration lock and move
# the hash of every study frozen before it. These constants are its settings.
WINDOWS: tuple[int | None, ...] = (6, 12, 24, None)   # months of memory; None keeps every earlier month
MIN_HISTORY = 6         # months before the first forecast when no fixed window fits the span
MIN_PER_CLASS = 5       # a period enters a factor's path with at least this many units of each class
MIN_PERIODS = 4         # with fewer periods in the path there is no mean and no spread to speak of
Z_CUT = 2.0             # a mean or a spread counts when it is this many standard errors from none
MAX_LEVELS = 12         # a text field with more distinct values gives no factors
MIN_LEVEL_SHARE = 0.02  # a level held by less than this share of the units (or by all but it) gives none

Reading = Literal["stable", "partly", "moving", "none", "few"]
Source = Literal["field", "rule", "question"]
READINGS: tuple[Reading, ...] = ("stable", "partly", "moving", "none", "few")
_READING: Mapping[tuple[bool, bool], Reading] = {
    (True, False): "stable", (True, True): "partly", (False, True): "moving", (False, False): "none"}
"""(does the mean count, does the spread count) -> the reading."""

# What a factor without information reads as something, from the two tests at Z_CUT: the mean test is
# two-sided, the spread test one-sided.
CHANCE = 1 - (1 - math.erfc(Z_CUT / math.sqrt(2))) * (1 - math.erfc(Z_CUT / math.sqrt(2)) / 2)


@dataclass(frozen=True)
class Factor:
    """One column over the units. x is nan where the unit gives no value (a null field, a refused
    score); `standardize` puts such a unit at the mean rank, where it tells nothing."""
    name: str
    source: Source
    x: np.ndarray
    filled_pct: float


@dataclass(frozen=True)
class Skipped:
    """What gave no factor, and why. Printed, so that "nothing stable" is never silent about what it
    did not look at."""
    name: str
    why: str


@dataclass(frozen=True)
class PeriodSlope:
    period: str
    units: int
    in_class: int
    b: float
    se: float


@dataclass(frozen=True)
class AcrossPeriods:
    """A slope path summed up. sigma_floor is the spread at which the heterogeneity statistic q is
    expected to reach Z_CUT: a smaller spread cannot be told from none with these periods."""
    periods: int
    mu: float
    mu_se: float
    tau2: float
    q: float
    q_z: float
    sigma_floor: float

    @property
    def sigma(self) -> float:
        return math.sqrt(self.tau2)

    @property
    def mu2(self) -> float:
        """mu^2 with its sampling noise taken out, as tau2 is: mu^2 alone is biased up by se(mu)^2."""
        return max(0.0, self.mu ** 2 - self.mu_se ** 2)

    @property
    def reading(self) -> Reading:
        """Two tests, four readings; no threshold on a ratio. `partly` says a part holds and a part
        moves, and stable_share says how much holds."""
        return _READING[abs(self.mu) >= Z_CUT * self.mu_se, self.q_z >= Z_CUT]

    @property
    def stable_share(self) -> float | None:
        """None when the reading is "none": the ratio of two noises."""
        total = self.mu2 + self.tau2
        return None if self.reading == "none" or total <= 0 else self.mu2 / total


@dataclass(frozen=True)
class Relearned:
    gain_pct: float    # 100 (CE of the re-learned base rate - CE of the re-learned slope) / H(Z), forecast units
    se_pct: float      # from the month-to-month spread of the gain: losses cluster in time
    months: int

    @property
    def starred(self) -> bool:
        return self.gain_pct > Z_CUT * self.se_pct


@dataclass(frozen=True)
class Walk:
    """The monthly re-learning walk of one class against the reference. steps[window] holds one
    (fit_on, forecast) pair of masks per forecast month. Every window forecasts the same months, the
    `months` here."""
    months: tuple[int, ...]
    steps: Mapping[str, tuple[tuple[np.ndarray, np.ndarray], ...]]


@dataclass(frozen=True)
class Versus:
    """One non-reference class against the reference class: which units are in either (`keep`, a mask
    over all surveyed units), and for those, y = 1 in the class, the month and the period."""
    name: str
    against: str
    keep: np.ndarray
    y: np.ndarray
    month: np.ndarray
    period: np.ndarray


# ---- the arithmetic: arrays in, numbers out ----------------------------------------------------------

def kappa(pi: float) -> float:
    """Nats per unit per b^2 of a standardized factor with a weak effect: I(Z;F) ~ kappa b^2, pi the
    share of the class (README.ja.md 7.3)."""
    return pi * (1 - pi) / 2


def standardize(x: np.ndarray) -> np.ndarray | None:
    """The factor as the survey reads it: midranks (tied values share their mean rank), units without
    a value at the mean rank, then mean 0 and sd 1. None when fewer than two distinct values.

    A rank depends on the value alone, never on the unit's position: the units are in time order, so
    breaking ties by position would plant a time trend in a factor with ties."""
    has = np.isfinite(x)
    values, inverse, counts = np.unique(x[has], return_inverse=True, return_counts=True)
    if len(values) < 2:
        return None
    mid = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    rank = np.full(len(x), mid.mean())
    rank[has] = mid
    return (rank - rank.mean()) / rank.std()


def _slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    """b of logit P(y = 1) = a + b x (Y.fit, weight 1), and its standard error from the Fisher
    information at the fit; None for the error when the information is singular."""
    B = Y.fit(x[:, None], y, np.ones(len(y)), 2)
    a, b = float(B[0, 0]), float(B[0, 1])
    p = 1 / (1 + np.exp(-np.clip(a + b * x, -Y.ETA_CLIP, Y.ETA_CLIP)))
    v = p * (1 - p)
    i00, i01, i11 = v.sum(), (v * x).sum(), (v * x * x).sum()
    det = i00 * i11 - i01 ** 2
    return b, (math.sqrt(i00 / det) if det > 1e-12 * i00 * i11 else None)


def period_slopes(x: np.ndarray, y: np.ndarray, period: np.ndarray) -> list[PeriodSlope]:
    """Per period in time order, the slope and its standard error. A period with fewer than
    MIN_PER_CLASS units of either class, or where x does not vary, is left out."""
    out = []
    for name in sorted(set(period.tolist())):
        m = period == name
        k = int(y[m].sum())
        if min(k, int(m.sum()) - k) < MIN_PER_CLASS or np.ptp(x[m]) == 0:
            continue
        b, se = _slope(x[m], y[m])
        if se is not None:
            out.append(PeriodSlope(name, int(m.sum()), k, b, se))
    return out


def _chi2_z(q: float, df: int) -> float:
    """A chi-square value with df degrees of freedom as a standard normal (Wilson-Hilferty)."""
    return ((q / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))


def _chi2_at(z: float, df: int) -> float:
    """The chi-square value with df degrees of freedom that _chi2_z maps to z."""
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def across_periods(path: Sequence[PeriodSlope]) -> AcrossPeriods | None:
    """DerSimonian-Laird over the path; None with fewer than MIN_PERIODS periods.

    w = 1 / se^2; q = sum w (b - fixed mean)^2; c = sum w - sum w^2 / sum w
    tau2 = max(0, (q - (k - 1)) / c); mu and mu_se with weights 1 / (se^2 + tau2)
    sigma_floor^2 = (the chi-square value at Z_CUT with k - 1 degrees - (k - 1)) / c, from E[q] = k - 1 + c tau2"""
    if len(path) < MIN_PERIODS:
        return None
    b, se = np.array([p.b for p in path]), np.array([p.se for p in path])
    w = 1 / se ** 2
    q = float((w * (b - (w * b).sum() / w.sum()) ** 2).sum())
    df, c = len(b) - 1, float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c)
    ws = 1 / (se ** 2 + tau2)
    return AcrossPeriods(len(b), float((ws * b).sum() / ws.sum()), 1 / math.sqrt(ws.sum()), tau2, q, _chi2_z(q, df),
                         math.sqrt(max(0.0, (_chi2_at(Z_CUT, df) - df) / c)))


def window_key(window: int | None) -> str:
    return "all" if window is None else str(window)


def walk(month: np.ndarray, y: np.ndarray, windows: Sequence[int | None], gap: int) -> Walk:
    """The forecast months and, per window, the units each month is fitted on. A month is forecast
    when it has units, when at least the longest fixed window of months lies before it after the `gap`
    months it takes an outcome to be known, and when every window's months hold both classes."""
    first, last = int(month.min()), int(month.max())
    history = max((w for w in windows if w is not None), default=MIN_HISTORY)
    months: list[int] = []
    steps: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {window_key(w): [] for w in windows}
    for m in range(first + history + gap, last + 1):
        forecast = month == m
        if not forecast.any():
            continue
        end = m - gap
        fits = [(month >= (first if w is None else end - w)) & (month < end) for w in windows]
        if not all(0 < y[f].sum() < f.sum() for f in fits):
            continue
        months.append(m)
        for w, f in zip(windows, fits):
            steps[window_key(w)].append((f, forecast))
    return Walk(tuple(months), {k: tuple(v) for k, v in steps.items()})


def _forecast_units(w: Walk) -> np.ndarray:
    return np.any([at for _, at in next(iter(w.steps.values()))], axis=0)


def _entropy_bits(pi: float) -> float:
    return 0.0 if pi <= 0 or pi >= 1 else -(pi * math.log2(pi) + (1 - pi) * math.log2(1 - pi))


def _loss_bits(y: np.ndarray, p: np.ndarray | float) -> float:
    p = np.clip(p, Y.CE_EPS, 1 - Y.CE_EPS)
    return float(-(y * np.log2(p) + (1 - y) * np.log2(1 - p)).sum())


def relearning(x: np.ndarray, y: np.ndarray, w: Walk) -> dict[str, Relearned | None]:
    """The prequential gain of one factor per window. Each month is forecast by a + b x fitted on its
    window and scored against the base rate of the same window; None with fewer than two months."""
    at_all = _forecast_units(w) if w.months else None
    h = 0.0 if at_all is None else int(at_all.sum()) * _entropy_bits(float(y[at_all].mean()))
    if len(w.months) < 2 or h <= 0:
        return {key: None for key in w.steps}
    out: dict[str, Relearned | None] = {}
    for key, steps in w.steps.items():
        d = []
        for fit_on, at in steps:
            B = Y.fit(x[fit_on][:, None], y[fit_on], np.ones(int(fit_on.sum())), 2)
            d.append(_loss_bits(y[at], float(y[fit_on].mean())) - _loss_bits(y[at], Y.predict_rest(B, x[at][:, None])[:, 0]))
        d = np.array(d)
        out[key] = Relearned(100 * d.sum() / h, 100 * math.sqrt(len(d) * d.var(ddof=1)) / h, len(d))
    return out


def no_information_pct(y: np.ndarray, w: Walk) -> dict[str, float | None]:
    """Where a factor without information is expected per window, in % of H(Z): fitting one slope on
    N units costs about 1 / (2N) nats per forecast unit, so the gain is -100 sum_m n_m / (2 N_m) over
    the forecast months, divided by H(Z) in nats of the forecast units."""
    if len(w.months) < 2:
        return {key: None for key in w.steps}
    at_all = _forecast_units(w)
    h_nats = int(at_all.sum()) * _entropy_bits(float(y[at_all].mean())) * math.log(2)
    return {key: (-100 * sum(float(at.sum()) / (2 * float(fit_on.sum())) for fit_on, at in steps) / h_nats if h_nats > 0 else None)
            for key, steps in w.steps.items()}


def _relearned(r: Relearned | None) -> dict[str, Any] | None:
    return None if r is None else {"gain_pct": Y.rnd(r.gain_pct, 3), "se_pct": Y.rnd(r.se_pct, 3), "months": r.months}


def measure(factor: Factor, vs: Versus, w: Walk, per_year: float) -> dict[str, Any]:
    """One factor against one class: its entry of the record, rounded as stored.

    info_pooled_pct = 100 kappa mu2 / H(Z) in nats: what one fixed slope can use; info_conditional_pct
    uses mu2 + sigma^2, what the slope of each period could use. nats_per_year = units per year x
    kappa mu2: what a year of data gathers about the fixed slope. With too few periods the reading is
    "few" and those numbers are None."""
    x = standardize(factor.x[vs.keep])
    path = period_slopes(x, vs.y, vs.period) if x is not None else []
    ap = across_periods(path)
    pi = float(vs.y.mean())
    k, h_nats = kappa(pi), _entropy_bits(pi) * math.log(2)
    out: dict[str, Any] = {"factor": factor.name, "source": factor.source, "filled_pct": factor.filled_pct,
                           "reading": ap.reading if ap else "few", "periods": len(path)}
    numbers: dict[str, Any] = dict.fromkeys(("mu", "mu_se", "sigma", "sigma_floor", "q", "q_z", "stable_share", "info_pooled_pct",
                                             "info_conditional_pct", "nats_per_year"))
    if ap and h_nats > 0:
        numbers.update(mu=Y.rnd(ap.mu, 4), mu_se=Y.rnd(ap.mu_se, 4), sigma=Y.rnd(ap.sigma, 4), sigma_floor=Y.rnd(ap.sigma_floor, 4),
                       q=Y.rnd(ap.q, 2), q_z=Y.rnd(ap.q_z, 2),
                       stable_share=None if ap.stable_share is None else Y.rnd(ap.stable_share, 3),
                       info_pooled_pct=Y.rnd(100 * k * ap.mu2 / h_nats, 3),
                       info_conditional_pct=Y.rnd(100 * k * (ap.mu2 + ap.tau2) / h_nats, 3),
                       nats_per_year=Y.rnd(per_year * k * ap.mu2, 2))
    out.update(numbers)
    learned = relearning(x, vs.y, w) if x is not None else {key: None for key in w.steps}
    out["relearning"] = {key: _relearned(r) for key, r in learned.items()}
    out["path"] = [{"period": p.period, "units": p.units, "in_class": p.in_class, "b": Y.rnd(p.b, 4), "se": Y.rnd(p.se, 4)}
                   for p in path]
    return out


def summary(factors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The counts of each reading next to what chance alone gives, the median floor of sigma, the
    largest information one fixed slope can use among the factors whose mean counts, and the best
    cell of the re-learning table."""
    count = {r: sum(f["reading"] == r for f in factors) for r in READINGS}
    floors = [f["sigma_floor"] for f in factors if f["sigma_floor"] is not None]
    holding = [f for f in factors if f["reading"] in ("stable", "partly") and f["info_pooled_pct"] is not None]
    top = max(holding, key=lambda f: f["info_pooled_pct"], default=None)
    cells = [(f, key, r) for f in factors for key, r in f["relearning"].items() if r is not None and r["gain_pct"] is not None]
    best = max(cells, key=lambda c: c[2]["gain_pct"], default=None)
    return {"factors": len(factors), **count, "by_chance": round(CHANCE * len(factors), 1),
            "sigma_floor": Y.rnd(float(np.median(floors)), 3) if floors else None,
            "largest_stable": top and {k: top[k] for k in ("factor", "reading", "info_pooled_pct", "nats_per_year")},
            "best_relearning": best and {"factor": best[0]["factor"], "window": best[1], "gain_pct": best[2]["gain_pct"],
                                         "se_pct": best[2]["se_pct"],
                                         "starred": best[2]["gain_pct"] > Z_CUT * best[2]["se_pct"]}}


def _order(f: Mapping[str, Any]) -> tuple:
    return READINGS.index(f["reading"]), -(f["info_pooled_pct"] or 0), -(f["info_conditional_pct"] or 0), f["factor"]


# ---- factors: what a task offers to be surveyed ------------------------------------------------------

def _kind_of(filled: Sequence[Any]) -> str:
    kinds = {"number" if isinstance(v, (bool, int, float)) else "text" if isinstance(v, str)
             else "list" if isinstance(v, list) else "object" for v in filled}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _field_factors(rows: Sequence[Row], names: Sequence[str]) -> tuple[list[Factor], list[Skipped]]:
    """One factor per numeric or true/false state field; one per level of a text field with at most
    MAX_LEVELS levels, each level held by MIN_LEVEL_SHARE to 1 - MIN_LEVEL_SHARE of the units (a
    field always filled with two levels gives one: the other is its complement). Anything else is
    Skipped with its reason."""
    factors, skipped = [], []
    for name in names:
        values = [r.state.get(name) for r in rows]
        filled = [v for v in values if v is not None]
        filled_pct = Y.rnd(100 * len(filled) / len(values), 1)
        kind = _kind_of(filled)
        if not filled:
            skipped.append(Skipped(name, "no values"))
        elif kind == "number":
            x = np.array([np.nan if v is None else float(v) for v in values])
            if standardize(x) is None:
                skipped.append(Skipped(name, "constant"))
            else:
                factors.append(Factor(name, "field", x, filled_pct))
        elif kind == "text":
            levels = sorted(set(filled))
            if len(levels) == 1 and len(filled) == len(values):
                skipped.append(Skipped(name, "constant"))
                continue
            if len(levels) > MAX_LEVELS:
                skipped.append(Skipped(name, f"{len(levels)} distinct texts"))
                continue
            if len(levels) == 2 and len(filled) == len(values):
                levels = levels[1:]
            made = [Factor(f"{name} = {level!r}", "field", np.array([float(v == level) for v in values]), filled_pct)
                    for level in levels]
            made = [f for f in made if MIN_LEVEL_SHARE <= f.x.mean() <= 1 - MIN_LEVEL_SHARE]
            factors += made
            if not made:
                skipped.append(Skipped(name, f"no level holds {100 * MIN_LEVEL_SHARE:.0f}% to {100 - 100 * MIN_LEVEL_SHARE:.0f}% of the units"))
        else:
            skipped.append(Skipped(name, {"list": "lists", "object": "objects", "mixed": "mixed types"}[kind]))
    return factors, skipped


def _prompt_factors(asker: ReaderPrompt, rows: Sequence[Row]) -> tuple[list[Factor], list[Skipped]]:
    """The machine's reading of a prompt, as factors. No decider is asked and no stored answer is read.

    A rules predicate: one factor per rule with an expr and, with two or more rules that all have one,
    one for the predicate itself. Any other question with an expr: the columns of KINDS[q.kind].factors
    from the option the machine's answer picks per unit. A rule or a question without an expr, or a
    column that does not vary, is Skipped."""
    data = [a.data for a in asker.asks(rows)]
    factors, skipped = [], []
    no_expr = "no expr: only a model can answer it (see `plab drift`)"
    for q in asker.questions:
        for r in q.rules:
            if r.expr is None:
                skipped.append(Skipped(f"rule {r.number}: {r.text}", no_expr))
            else:
                factors.append(Factor(f"rule {r.number}: {r.text}", "rule", np.array([float(r.expr.applies(d)) for d in data]), 100.0))
        if len(q.rules) == 1:
            continue
        if not q.machine_ready:
            skipped.append(Skipped(q.name, "a rule without expr" if q.rules else no_expr))
            continue
        kind = KINDS[q.kind]
        picked = np.array([kind.chosen(machine_answer(q, d)) for d in data], int)
        filled_pct = Y.rnd(100 * float((picked >= 0).mean()), 1)
        factors += [Factor(f"{q.name}: {label}", "question", x, filled_pct) for label, x in kind.factors(q, picked)]
    varies = [standardize(f.x) is not None for f in factors]
    return ([f for f, v in zip(factors, varies) if v],
            skipped + [Skipped(f.name, "constant") for f, v in zip(factors, varies) if not v])


# ---- the shell ---------------------------------------------------------------------------------------

def _read(study: Study, every_split: bool) -> tuple[tuple[Row, ...], tuple[str, ...], dict[str, str]]:
    """The rows the survey may read now, in time order, the splits they came from and the sha256 of
    each split file. The Lock is asked for here and nowhere else, so no caller can hand in one made
    for another command; Study.rows refuses the evaluation and test rows without it."""
    lock = protocol.enter(study, "survey") if every_split else None
    splits = tuple(study.task.splits) if every_split else ("train",)
    read = [study.rows(s, lock) for s in splits]
    return tuple(r for s in read for r in s.rows), splits, {s.split: s.sha256 for s in read}


def _month(t: Any) -> int:
    return t.year * 12 + t.month - 1


def _label(month: int) -> str:
    return f"{month // 12}-{month % 12 + 1:02d}"


def survey(study: Study, asker: ReaderPrompt | None, *, by: str, gap: int, every_split: bool, agent: AgentName) -> dict[str, Any]:
    """The whole survey. Factors come from the state fields, or from `asker` when one is given.

    The record holds no clock time: the same rows give the same numbers. A survey that read more than
    the training split is appended to the agent's `surveys` log, as every look at those rows is."""
    if gap < 0:
        raise PlabError("--gap is the number of whole months before an outcome is known: 0 or more")
    rows, splits, sha = _read(study, every_split)
    factors, skipped = _prompt_factors(asker, rows) if asker else _field_factors(rows, study.all_fields())
    if not factors:
        raise PlabError("nothing to survey: " + ("no rule or question of this prompt has an expr the machine can answer, "
                                                 "or none of their answers varies" if asker else "no state field gives a factor")
                        + "".join(f"; {s.name}: {s.why}" for s in skipped[:6]))
    task, exploratory = study.task, splits != ("train",)
    z = np.array([r.z for r in rows])
    month = np.array([_month(r.t) for r in rows])
    period = np.array([period_key(r.t, by) for r in rows])
    span = int(month.max() - month.min() + 1)
    windows = tuple(w for w in WINDOWS if w is None or 2 * w <= span)
    blocks = []
    for c in range(1, task.outcome.K):
        keep = (z == 0) | (z == c)
        vs = Versus(task.outcome.classes[c], task.outcome.classes[0], keep, (z[keep] == c).astype(int), month[keep], period[keep])
        w = walk(vs.month, vs.y, windows, gap)
        per_year = 12 * len(vs.y) / span
        measured = sorted((measure(f, vs, w, per_year) for f in factors), key=_order)
        pi = float(vs.y.mean())
        blocks.append({"class": vs.name, "against": vs.against, "units": len(vs.y), "units_in_class": int(vs.y.sum()),
                       "pct": Y.rnd(100 * pi, 2), "h_bits": Y.rnd(_entropy_bits(pi), 4),
                       "predicted_months": {"first": _label(w.months[0]), "last": _label(w.months[-1]), "count": len(w.months)}
                       if w.months else None,
                       "no_information_pct": {k: Y.rnd(v, 3) for k, v in no_information_pct(vs.y, w).items()},
                       "summary": summary(measured), "factors": measured})
    record = {"task": task.name, "harness_version": __version__,
              "factors_from": {"prompt": asker.name, "prompt_id": asker.prompt_id, "file": asker.file} if asker else "fields",
              "splits": list(splits), "exploratory": exploratory, "data_sha256": sha, "by": by, "gap_months": gap,
              "windows_months": list(windows), "units": len(rows), "first": _label(int(month.min())), "last": _label(int(month.max())),
              "months": span, "periods": len(set(period.tolist())), "units_per_month": Y.rnd(len(rows) / span, 1),
              "skipped": [{"factor": s.name, "why": s.why} for s in skipped], "by_class": blocks}
    return study.layout.append_log(agent, "surveys", record) if exploratory else record
