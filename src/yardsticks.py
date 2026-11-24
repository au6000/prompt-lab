"""Yardsticks: arrays and labels go in, numbers come out. Pure numpy; nothing here reads a file,
a prompt or a decision model.

Vocabulary (the theory at the top of README.ja.md):
  Z   the outcome, K >= 2 classes. Class 0 is the reference class. With K == 2, class 1 is the
      positive class.
  w   unit weights (1 everywhere except in a down-sampled training split).
  V   the family of the recalibrated gain: softmax(a + B x) over the encoded answers x, with
      class 0 as the reference.
  s   the forecast probability of the positive class of a two-class outcome.

Reference form. A probability vector over n options (an answer) or K classes (a forecast) is held
as its n-1 (K-1) non-reference entries, called `rest`. The reference entry is always the remainder
1 - sum(rest). The entropy, the cross-entropy and the fit of V add the non-reference terms first
and the remainder last, and make one matrix-vector product per class. With a single non-reference
entry that is, operation for operation, the two-class arithmetic of version 0.1.0, so its numbers
come out of the general code and there is no two-class twin to keep in step.
tests/test_yardsticks.py holds this with exact float equality against tests/golden/v01_metrics.py;
keep the order of operations when editing.

The information yardsticks (entropy_bits, ce_bits, gain_pct, recalibrated_gain_pct) take any K and
any number of questions. The use yardsticks (wauc, binned_mi_pct, threshold_for_best, profit_xfit,
decision, final_summary, rule_table) read s or the positive class, so they exist for a two-class
outcome only; they are not special cases of a general quantity.

Sign. cv_gain_pct and mi_pct are sign-blind: a reversed answer scores the same, because a slope or
the bin labels absorb the direction. The others are sign-aware: they score in a direction set
beforehand, the one the prompt's forecast states or, for a fitted forecast and transfer_gain_pct,
the one V took on other units (other folds of the training split, or the training split itself).
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np

# Every value below is part of the numbers already on record: changing one changes every result.
LOGIT_EPS = 1e-4       # answers are clipped to [eps, 1 - eps] before a log-ratio
CE_EPS = 1e-6          # forecasts are clipped before the log of a cross-entropy
H_EPS = 1e-12          # class shares are clipped before the entropy
RIDGE = 1e-3           # on the slopes of V, never on the intercepts
NEWTON_MAX_IT = 50
NEWTON_TOL = 1e-8      # the fit stops when its largest step is below this
ETA_CLIP = 30
HESS_JITTER = 1e-9
CV_SEED_BASE = 100     # repetition r draws its folds from default_rng(100 + r)
MI_BINS = 5
MI_PERM_SEED = 5
TEST_PERMS = 2000
TEST_SEED = 9
XFIT_STEP = 0.05       # the cross-fit tries skipping 0%, 5%, ..., 100% of the weight


@dataclass(frozen=True)
class Labels:
    """What the yardsticks need to know about the units, aligned by position.

    z       class index of each unit; 0 is the reference class
    K       number of outcome classes
    w       unit weights
    groups  cross-validation group keys; units of one group share a fold
    half    1 or 2 for the two training halves, 0 elsewhere
    value   the outcome value of each unit, or None when the task has none
    """
    z: np.ndarray
    K: int
    w: np.ndarray
    groups: np.ndarray
    half: np.ndarray
    value: np.ndarray | None

    @property
    def y(self) -> np.ndarray:
        """1.0 where the unit is in class 1, the positive class of a two-class outcome."""
        return (self.z == 1).astype(float)


@dataclass(frozen=True)
class Forecast:
    """Class probabilities per unit in reference form, (N, K-1), and where they come from.

    fitted is False when they are the decision model's own probabilities, so the as-is gain exists,
    and True when V was fitted on training answers. out_of_fold is True for a fitted forecast on the
    training split, where each unit is forecast by V fitted on the other folds.
    """
    rest: np.ndarray
    fitted: bool
    out_of_fold: bool = False

    @property
    def s(self) -> np.ndarray:
        return self.rest[:, 0]


def _cap_rest(rest: np.ndarray, eps: float) -> np.ndarray:
    """Clip the entries of `rest`, (N, m), to [eps, 1 - eps]. A row that then sums above 1 - eps is
    scaled down to it, so the remainder never falls below about eps."""
    R = np.clip(rest, eps, 1 - eps)
    return R * np.minimum(1.0, (1 - eps) / R.sum(axis=1))[:, None]


def log_ratios(rest: np.ndarray) -> np.ndarray:
    """The recalibration features of one question, (N, m): log(p_j / p_0) of each non-reference
    option j against the remainder p_0, the probabilities capped at LOGIT_EPS first.

    A yes/no answer is rest = [P(yes)] and gives logit(P(yes)). A choice is the probabilities of
    its options after the first; an answer that named an option without probabilities is that
    option's point mass. A score is read by its mean: rest = [mean level / (levels - 1)].
    """
    R = _cap_rest(rest, LOGIT_EPS)
    p0 = 1 - R.sum(axis=1)
    return np.column_stack([np.log(R[:, j] / p0) for j in range(R.shape[1])])


def forecast_rest(rest: np.ndarray, option_class: Sequence[int], K: int) -> np.ndarray:
    """The decision model's own forecast of the outcome, (N, K-1), from one question's answers.

    option_class[j] is the class that option j forecasts. A class gets the sum of its options'
    probabilities, added in option order; option 0 holds the remainder of `rest`.
    """
    options = [1 - rest.sum(axis=1), *rest.T]
    out = np.zeros((len(rest), K - 1))
    for p, c in zip(options, option_class, strict=True):
        if c:
            out[:, c - 1] += p
    return out


def entropy_bits(labels: Labels) -> float:
    """Plug-in H(Z) in bits, from the weighted class shares."""
    W = labels.w.sum()
    shares = np.array([[(labels.w * (labels.z == c)).sum() / W for c in range(1, labels.K)]])
    rest = _cap_rest(shares, H_EPS)[0]
    return -sum(p * np.log2(p) for p in (*rest, 1 - rest.sum()))


def ce_bits(rest: np.ndarray, labels: Labels) -> float:
    """Weighted mean cross-entropy in bits of a forecast in reference form."""
    R = _cap_rest(rest, CE_EPS)
    terms = 0.0
    for c in range(1, labels.K):
        terms = terms + (labels.z == c) * np.log2(R[:, c - 1])
    terms = terms + (labels.z == 0) * np.log2(1 - R.sum(axis=1))
    return float(-(labels.w * terms).sum() / labels.w.sum())


def gain_pct(rest: np.ndarray, labels: Labels) -> float:
    """100 (H(Z) - CE) / H(Z) of a forecast: the share of the outcome's entropy it removes.

    Of the decision model's own probabilities this is the as-is gain. Nothing is fitted and no
    constant forecast is in the family, so it can be negative however many units there are; what it
    loses to the recalibrated gain is the cost of trusting the raw probabilities.
    """
    h = entropy_bits(labels)
    return 100 * (h - ce_bits(rest, labels)) / h


def _with_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _softmax_rest(eta: list[np.ndarray]) -> np.ndarray:
    """Class probabilities in reference form, (N, K-1), from the clipped score of each
    non-reference class; the reference class scores 0."""
    out = np.empty((len(eta[0]), len(eta)))
    for c, e in enumerate(eta):
        den = 1
        for c2, e2 in enumerate(eta):
            if c2 != c:
                den = den + np.exp(e2 - e)
        out[:, c] = 1 / (den + np.exp(-e))
    return out


def _probabilities(B: np.ndarray, X1: np.ndarray) -> np.ndarray:
    return _softmax_rest([np.clip(X1 @ b, -ETA_CLIP, ETA_CLIP) for b in B])


def predict_rest(B: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Class probabilities of V with coefficients B at the predictors X, (N, K-1)."""
    return _probabilities(B, _with_intercept(X))


def fit(X: np.ndarray, z: np.ndarray, w: np.ndarray, K: int) -> np.ndarray:
    """Fit V by weighted Newton steps. Returns B, (K-1, d+1): one row per non-reference class,
    the intercept first. A small ridge holds the slopes; the intercepts are free."""
    X1 = _with_intercept(X)
    m, d1 = K - 1, X1.shape[1]
    in_class = [(z == c).astype(float) for c in range(1, K)]
    ridge = RIDGE * np.diag(np.r_[0, np.ones(d1 - 1)])
    B = np.zeros((m, d1))
    for _ in range(NEWTON_MAX_IT):
        P = _probabilities(B, X1)
        g = np.concatenate([X1.T @ (w * (P[:, c] - in_class[c])) + RIDGE * np.r_[0, B[c, 1:]] for c in range(m)])
        H = np.zeros((m * d1, m * d1))
        for c in range(m):
            for c2 in range(m):
                block = (X1 * (w * P[:, c] * (float(c == c2) - P[:, c2]))[:, None]).T @ X1
                H[c * d1:(c + 1) * d1, c2 * d1:(c2 + 1) * d1] = block + ridge if c == c2 else block
        step = np.linalg.solve(H + HESS_JITTER * np.eye(m * d1), g)
        B = B - step.reshape(m, d1)
        if np.abs(step).max() < NEWTON_TOL:
            break
    return B


def _folds(groups: np.ndarray, cv_folds: int, seed: int) -> np.ndarray:
    """Fold index per unit. A whole group goes to one fold: units close in time share an episode,
    and splitting it across folds would leak."""
    g = np.unique(groups)
    fold_of_group = np.random.default_rng(seed).permutation(len(g)) % cv_folds
    return fold_of_group[np.searchsorted(g, groups)]


def _held_out(labels: Labels, cv_folds: int, cv_reps: int) -> Iterator[np.ndarray]:
    """The mask of each fold that holds units, over cv_reps splits into cv_folds folds."""
    for r in range(cv_reps):
        f = _folds(labels.groups, cv_folds, CV_SEED_BASE + r)
        for j in range(cv_folds):
            te = f == j
            if te.any():
                yield te


def _out_of_fold(labels: Labels, cv_folds: int, cv_reps: int,
                 fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """Out-of-fold class probabilities, (N, K-1): the mean over the repetitions.
    fit_predict(fit_on, predict_at) fits on the units of the first mask and predicts at the second."""
    P = np.zeros((len(labels.z), labels.K - 1))
    for te in _held_out(labels, cv_folds, cv_reps):
        P[te] += fit_predict(~te, te) / cv_reps
    return P


def _fit_predict(X: np.ndarray, labels: Labels) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    return lambda tr, te: predict_rest(fit(X[tr], labels.z[tr], labels.w[tr], labels.K), X[te])


def cv_predict(X: np.ndarray, labels: Labels, cv_folds: int, cv_reps: int) -> np.ndarray:
    """Out-of-fold class probabilities of V, (N, K-1)."""
    return _out_of_fold(labels, cv_folds, cv_reps, _fit_predict(X, labels))


def recalibrated_gain_pct(X: np.ndarray, labels: Labels, cv_folds: int, cv_reps: int) -> float:
    """cv_gain_pct: the gain of V's out-of-fold forecast from the encoded answers X, (N, d).

    An estimate of the V-information of the answers about the outcome. V holds the constant
    forecast, so with endless units it is never below zero; on a finite sample answers without
    information score below zero, which is why it is read against a zero point. Sign-blind.
    """
    return gain_pct(cv_predict(X, labels, cv_folds, cv_reps), labels)


def _along(a: np.ndarray, strength: float, E: np.ndarray) -> np.ndarray:
    return _softmax_rest([np.clip(a[c] + strength * E[:, c], -ETA_CLIP, ETA_CLIP) for c in range(len(a))])


def fit_along(E: np.ndarray, z: np.ndarray, w: np.ndarray, K: int) -> tuple[np.ndarray, float]:
    """Fit softmax(a + c E) with c >= 0 by weighted Newton steps. E, (N, K-1), holds each unit's score
    for every non-reference class under a V fitted elsewhere. Returns the intercepts a, (K-1,), which
    are free, and the strength c, which takes the ridge of the slopes.

    The likelihood is concave in (a, c), so when its best c is negative the best c >= 0 is c = 0: the
    class shares, as if there were no answers."""
    m = K - 1
    in_class = np.column_stack([(z == c).astype(float) for c in range(1, K)])

    def newton(free: int) -> np.ndarray:
        t = np.zeros(m + 1)
        for _ in range(NEWTON_MAX_IT):
            P = _along(t[:m], t[m], E)
            wP, wR = w[:, None] * P, w[:, None] * (P - in_class)
            Pe = (P * E).sum(axis=1)
            g = np.r_[wR.sum(axis=0), (wR * E).sum() + RIDGE * t[m]]
            H = np.empty((m + 1, m + 1))
            H[:m, :m] = np.diag(wP.sum(axis=0)) - wP.T @ P
            H[:m, m] = H[m, :m] = (wP * (E - Pe[:, None])).sum(axis=0)
            H[m, m] = (w * ((P * E ** 2).sum(axis=1) - Pe ** 2)).sum() + RIDGE
            step = np.linalg.solve(H[:free, :free] + HESS_JITTER * np.eye(free), g[:free])
            t[:free] = t[:free] - step
            if np.abs(step).max() < NEWTON_TOL:
                break
        return t

    t = newton(m + 1)
    if t[m] < 0:
        t = newton(m)
    return t[:m], float(t[m])


def transfer_gain_pct(X: np.ndarray, B: np.ndarray, labels: Labels, cv_folds: int, cv_reps: int) -> float:
    """transfer_gain_pct: the gain on this split of V kept as it was fitted on the training split.

    B is V fitted on the training answers (`fit`). Each unit's class scores E = X B' (intercepts left
    out) stay as trained; only the intercepts and a strength c >= 0 are fitted here, out of fold as in
    cv_gain_pct. An answer whose relation to the outcome turned around since training scores at or
    below zero; one whose relation only weakened keeps a positive gain; a change of the class shares
    costs nothing. It fits one number more than the constant forecast, whatever the number of
    columns. Sign-aware.
    """
    E = X @ B[:, 1:].T

    def fit_predict(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        a, c = fit_along(E[tr], labels.z[tr], labels.w[tr], labels.K)
        return _along(a, c, E[te])

    return gain_pct(_out_of_fold(labels, cv_folds, cv_reps, fit_predict), labels)


def cv_auc(X: np.ndarray, labels: Labels, cv_folds: int, cv_reps: int) -> tuple[float | None, float | None, float | None]:
    """The AUC of V's out-of-fold forecast of a two-class outcome, for the whole split and for each
    training half. It is taken within each fold, where every unit is ranked by the same fit, and the
    folds and repetitions are averaged by their weight of positive-negative pairs.

    Pooled over folds the AUC of a weak answer falls below 0.5: a fold with more positives than the
    rest is predicted by a fit that saw fewer, so its units are ranked low. None for a part without
    a pair."""
    y, w, fit_predict = labels.y, labels.w, _fit_predict(X, labels)
    num, den = np.zeros(3), np.zeros(3)
    for te in _held_out(labels, cv_folds, cv_reps):
        s = fit_predict(~te, te)[:, 0]
        for part, m in enumerate((np.ones(int(te.sum()), bool), labels.half[te] == 1, labels.half[te] == 2)):
            yy, ww = y[te][m], w[te][m]
            pairs = (ww * yy).sum() * (ww * (1 - yy)).sum()
            if pairs > 0:
                num[part] += wauc(s[m], yy, ww) * pairs
                den[part] += pairs
    return tuple(float(num[p] / den[p]) if den[p] > 0 else None for p in range(3))


def wauc(s: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUC, ties counted as half; nan when a class has no weight."""
    o = np.argsort(s, kind="mergesort")
    s, y, w = s[o], y[o], w[o]
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cw = np.zeros(len(cnt))
    cp = np.zeros(len(cnt))
    np.add.at(cw, inv, w * (1 - y))
    np.add.at(cp, inv, w * y)
    below = np.cumsum(cw) - cw
    den = cw.sum() * cp.sum()
    if den <= 0:
        return float("nan")
    return float(((below + cw / 2) * cp).sum() / den)


def _bins(s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """MI_BINS bins of equal weight. Equal values never straddle two bins: they go to the lower."""
    o = np.argsort(s, kind="mergesort")
    cw = np.cumsum(w[o]) / w.sum()
    out = np.empty(len(s), int)
    out[o] = np.minimum((cw * MI_BINS - 1e-9).astype(int), MI_BINS - 1)
    _, inv = np.unique(s, return_inverse=True)
    first = np.full(inv.max() + 1, MI_BINS)
    np.minimum.at(first, inv, out)
    return first[inv]


def _mi_bits(b: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted plug-in mutual information between the bin and the outcome, in bits."""
    W = w.sum()
    py = (w * y).sum() / W
    nb = int(b.max()) + 1
    wk = np.bincount(b, weights=w, minlength=nb)
    w1 = np.bincount(b, weights=w * (y == 1), minlength=nb)
    w0 = np.bincount(b, weights=w * (y == 0), minlength=nb)
    mi = 0.0
    for k in range(nb):
        if wk[k] <= 0:
            continue
        pk = wk[k] / W
        for pj, pyy in ((w1[k] / W, py), (w0[k] / W, 1 - py)):
            if pj > 0:
                mi += pj * np.log2(pj / (pk * pyy))
    return mi


def _mi_against_shuffles(s: np.ndarray, labels: Labels, nperm: int,
                         rng: np.random.Generator) -> tuple[float, float, np.ndarray]:
    y, w = labels.y, labels.w
    b = _bins(s, w)
    mi = _mi_bits(b, y, w)
    null = np.array([_mi_bits(b[rng.permutation(len(b))], y, w) for _ in range(nperm)])
    return 100 * (mi - null.mean()) / entropy_bits(labels), float(((null >= mi).sum() + 1) / (nperm + 1)), b


def binned_mi_pct(s: np.ndarray, labels: Labels, mi_perms: int) -> tuple[float, float, np.ndarray]:
    """mi_pct, its p and the bins: the plug-in mutual information between MI_BINS equal-weight
    bins of s and the outcome, minus its mean over mi_perms shuffles of the bins, in % of H(Z).

    The check that fits no model. The plug-in estimate is biased upward, hence the shuffles.
    Sign-blind.
    """
    return _mi_against_shuffles(s, labels, mi_perms, np.random.default_rng(MI_PERM_SEED))


def threshold_for_best(s: np.ndarray, w: np.ndarray, value: np.ndarray) -> float:
    """Among skipping the top 0%, 5%, ..., 100% of the weight, the threshold that leaves the best
    value per unit (skip when s >= threshold). inf skips nothing, -inf everything."""
    o = np.argsort(-s)
    cw = np.cumsum(w[o]) / w.sum()
    best_v, best_t = None, np.inf
    for fr in np.arange(0, 1.0001, XFIT_STEP):
        if fr < 1e-9:
            thr = np.inf
        elif fr > 1 - 1e-9:
            thr = -np.inf
        else:
            thr = s[o][min(np.searchsorted(cw, fr), len(o) - 1)]
        keep = s < thr
        v = (w[keep] * value[keep]).sum() / w.sum()
        if best_v is None or v > best_v + 1e-12:
            best_v, best_t = v, thr
    return best_t


def profit_xfit(s: np.ndarray, labels: Labels) -> dict | None:
    """Cross-fit over the training halves: the best skip share of one half, applied as a threshold
    to the other, both ways. None unless both halves have units."""
    y, w, value, half = labels.y, labels.w, labels.value, labels.half
    tot = kept = kneg = val = 0.0
    for a, b in ((1, 2), (2, 1)):
        ma, mb = half == a, half == b
        if not ma.any() or not mb.any():
            return None
        keep = s[mb] < threshold_for_best(s[ma], w[ma], value[ma])
        wb = w[mb]
        tot += wb.sum()
        kept += wb[keep].sum()
        kneg += (wb[keep] * (1 - y[mb][keep])).sum()
        val += (wb[keep] * value[mb][keep]).sum()
    return {"value_per_1000": round(1000 * val / tot, 1), "keep_frac": round(kept / tot, 3),
            "kept_neg_pct": round(100 * kneg / kept, 2) if kept > 0 else None}


def decision(s: np.ndarray, labels: Labels, threshold: float) -> dict:
    """The veto use: skip a unit when s >= threshold. What is kept and skipped, and the value per
    1,000 units next to taking everything."""
    y, w, value = labels.y, labels.w, labels.value
    kp = s < threshold
    W = w.sum()
    kept_pos = (w[kp] * y[kp]).sum() / w[kp].sum() if kp.any() else None
    kept_neg = (w[kp] * (1 - y[kp])).sum() / w[kp].sum() if kp.any() else None
    skip_pos = (w[~kp] * y[~kp]).sum() / w[~kp].sum() if (~kp).any() else None
    v = 1000 * (w[kp] * value[kp]).sum() / W
    allin = 1000 * (w * value).sum() / W
    return {"threshold": threshold,
            "skip_frac": round(float(w[~kp].sum() / W), 3),
            "kept_pos_pct": rnd(100 * kept_pos, 2) if kept_pos is not None else None,
            "kept_neg_pct": rnd(100 * kept_neg, 2) if kept_neg is not None else None,
            "skipped_pos_pct": rnd(100 * skip_pos, 2) if skip_pos is not None else None,
            "lift": rnd(skip_pos / kept_pos, 3) if (kept_pos and skip_pos is not None) else None,
            "value_per_1000": round(float(v), 1),
            "all_in_value_per_1000": round(float(allin), 1),
            "gain_vs_all_in_per_1000": round(float(v - allin), 1)}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - r) / d), float((c + r) / d))


def final_summary(s: np.ndarray, labels: Labels, threshold: float) -> dict:
    """The one-shot test table, every unit counted once whatever its weight: units taken, their
    rates with Wilson 95% intervals, value per unit and in total, the AUC with a one-sided
    permutation p, and the binned MI against its shuffles."""
    y, value = labels.y, labels.value
    unweighted = replace(labels, w=np.ones(len(y)))
    keep = s < threshold
    n = int(keep.sum())
    kpos = int((keep & (y == 1)).sum())
    lo, hi = _wilson(n - kpos, n)
    plo, phi = _wilson(kpos, n)
    out = {"rows": int(len(y)), "taken": n, "skipped": int(len(y) - n),
           "skip_frac": round(1 - n / len(y), 3) if len(y) else None,
           "taken_pos": kpos, "taken_neg": n - kpos,
           "taken_neg_pct": round(100 * (n - kpos) / n, 2) if n else None,
           "taken_neg_ci95": [round(100 * lo, 1), round(100 * hi, 1)] if n else None,
           "taken_pos_pct": round(100 * kpos / n, 2) if n else None,
           "taken_pos_ci95": [round(100 * plo, 1), round(100 * phi, 1)] if n else None,
           "value_per_taken": round(float(value[keep].mean()), 3) if n else None,
           "value_total": round(float(value[keep].sum()), 1)}
    if len(np.unique(s)) > 1 and 0 < y.sum() < len(y):
        rng = np.random.default_rng(TEST_SEED)
        w = unweighted.w
        a = wauc(s, y, w)
        pa = (sum(wauc(s, rng.permutation(y), w) >= a for _ in range(TEST_PERMS)) + 1) / (TEST_PERMS + 1)
        mi, mi_p, _ = _mi_against_shuffles(s, unweighted, TEST_PERMS, rng)
        out.update({"auc": round(a, 4), "auc_p_one_sided": round(float(pa), 4),
                    "mi_pct": round(mi, 3), "mi_p": round(mi_p, 4)})
    return out


def percentile(null_values: Sequence[float | None], x: float | None) -> float | None:
    """Where x falls among the zero-point draws, in percent; ties count half."""
    v = np.asarray([z for z in null_values if z is not None and np.isfinite(z)], float)
    if x is None or not len(v) or not np.isfinite(x):
        return None
    return round(100 * (float((v < x).sum()) + 0.5 * float((v == x).sum())) / len(v), 1)


def rnd(x: float | None, k: int) -> float | None:
    """round(), with None for a missing or non-finite value so that a record stays valid JSON."""
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), k)


def rule_table(hits: Sequence[np.ndarray | None], labels: Labels) -> list[dict]:
    """Per rule of a two-class task: the share of the weight it hits, and the positive rate where
    it hits and where it does not. hits[j] is None for a rule without a machine form."""
    y, w = labels.y, labels.w
    out = []
    for number, h in enumerate(hits, 1):
        if h is None:
            out.append({"rule": number, "expr": None})
            continue
        hit = (w[h] * y[h]).sum() / w[h].sum() if h.any() else None
        rest = (w[~h] * y[~h]).sum() / w[~h].sum() if (~h).any() else None
        out.append({"rule": number, "hit_frac": rnd(w[h].sum() / w.sum(), 3),
                    "pos_pct_hit": rnd(100 * hit, 2) if hit is not None else None,
                    "pos_pct_rest": rnd(100 * rest, 2) if rest is not None else None,
                    "lift": rnd(hit / rest, 2) if (hit is not None and rest) else None})
    return out


def score(X: np.ndarray, labels: Labels, *, cv_folds: int, cv_reps: int, mi_perms: int,
          forecast: Forecast | None = None, threshold: float | None = None, trained: np.ndarray | None = None) -> dict:
    """The yardsticks of one measurement, rounded as they are stored.

    Always cv_gain_pct, from the encoded answers X. With `trained`, V fitted on the training answers
    (when this is another split), transfer_gain_pct. With a forecast that was not fitted,
    asis_gain_pct. With a forecast of a two-class outcome, auc, auc_h1, auc_h2, mi_pct, mi_p and
    pos_by_quintile_pct; and, when the units carry a value, xfit on the training halves and
    decision at `threshold`. An out-of-fold forecast ranks the units of each fold by a fit of its
    own, so its AUC comes from within the folds (`cv_auc`) and mi_pct, mi_p, pos_by_quintile_pct and
    xfit, which compare units across folds, are left out (None, or no xfit).
    """
    gains = {"cv_gain_pct": round(recalibrated_gain_pct(X, labels, cv_folds, cv_reps), 3)}
    if trained is not None:
        gains["transfer_gain_pct"] = round(transfer_gain_pct(X, trained, labels, cv_folds, cv_reps), 3)
    if forecast is None:
        return gains
    if not forecast.fitted:
        gains["asis_gain_pct"] = round(gain_pct(forecast.rest, labels), 3)
    if labels.K != 2:
        return gains
    s, y, w, half = forecast.s, labels.y, labels.w, labels.half
    if forecast.out_of_fold:
        auc, auc_h1, auc_h2 = (rnd(x, 4) for x in cv_auc(X, labels, cv_folds, cv_reps))
        mi = mi_p = quintiles = None
    else:
        mi_bits, mi_shuffled_p, bins = binned_mi_pct(s, labels, mi_perms)
        auc = rnd(wauc(s, y, w), 4)
        auc_h1, auc_h2 = (rnd(wauc(s[m], y[m], w[m]), 4) if m.any() else None for m in (half == 1, half == 2))
        mi, mi_p = round(mi_bits, 3), round(mi_shuffled_p, 4)
        quintiles = [round(100 * float((w[m] * y[m]).sum() / w[m].sum()), 2) if m.any() else None
                     for m in (bins == q for q in range(MI_BINS))]
    block = {"auc": auc, "auc_h1": auc_h1, "auc_h2": auc_h2, "mi_pct": mi, "mi_p": mi_p, **gains, "pos_by_quintile_pct": quintiles}
    if labels.value is not None:
        if (half > 0).any() and not forecast.out_of_fold:
            block["xfit"] = profit_xfit(s, labels)
        if threshold is not None:
            block["decision"] = decision(s, labels, threshold)
    return block
