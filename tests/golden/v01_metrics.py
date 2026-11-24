"""Yardsticks. Every prompt is measured with the same functions.

Notation: s = the answer turned into "higher means more likely positive" (for a veto prompt: the
probability of skipping), y = label (1 = positive, the outcome a veto should avoid), w = row weight
(1 everywhere except the down-sampled training set), value = the outcome value of the row (P&L).
"""
import numpy as np

EPS = 1e-4


def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def wauc(s, y, w):
    """Weighted AUC, ties counted as half."""
    s = np.asarray(s, float); y = np.asarray(y, float); w = np.asarray(w, float)
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cw = np.zeros(len(cnt)); cp = np.zeros(len(cnt))
    np.add.at(cw, inv, w * (1 - y)); np.add.at(cp, inv, w * y)
    below = np.cumsum(cw) - cw
    den = cw.sum() * cp.sum()
    if den <= 0:
        return float("nan")
    return float(((below + cw / 2) * cp).sum() / den)


def wbins(s, w, K=5):
    """K bins of equal weight; equal values never straddle two bins (they go to the first one)."""
    o = np.argsort(s, kind="mergesort")
    cw = np.cumsum(w[o]) / w.sum()
    b = np.minimum((cw * K - 1e-9).astype(int), K - 1)
    out = np.empty(len(s), int); out[o] = b
    _, inv = np.unique(s, return_inverse=True)
    first = np.full(inv.max() + 1, K); np.minimum.at(first, inv, out)
    return first[inv]


def wmi(b, y, w):
    """Weighted mutual information I(Y; bin) in bits."""
    W = w.sum(); py = (w * y).sum() / W
    K = int(b.max()) + 1
    wk = np.bincount(b, weights=w, minlength=K)
    w1 = np.bincount(b, weights=w * (y == 1), minlength=K)
    w0 = np.bincount(b, weights=w * (y == 0), minlength=K)
    I = 0.0
    for k in range(K):
        if wk[k] <= 0:
            continue
        pk = wk[k] / W
        for pj, pyy in ((w1[k] / W, py), (w0[k] / W, 1 - py)):
            if pj > 0:
                I += pj * np.log2(pj / (pk * pyy))
    return I


def fit_lr(X, y, w, l2=1e-3, it=50):
    """Weighted logistic regression by Newton steps (tiny ridge on the slopes)."""
    X1 = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X1.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X1 @ b, -30, 30)))
        g = X1.T @ (w * (p - y)) + l2 * np.r_[0, b[1:]]
        H = (X1 * (w * p * (1 - p))[:, None]).T @ X1 + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b


def pred_lr(b, X):
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ b, -30, 30)))


def folds(y, k, seed, groups=None):
    """Fold labels. With groups (e.g. calendar months) every group goes to one fold, because
    nearby rows share the same losing episode and would otherwise leak across folds."""
    rng = np.random.default_rng(seed)
    if groups is not None:
        g = np.unique(groups); a = rng.permutation(len(g)) % k
        return a[np.searchsorted(g, groups)]
    f = np.empty(len(y), int)
    for c in (0, 1):
        idx = np.flatnonzero(y == c); rng.shuffle(idx); f[idx] = np.arange(len(idx)) % k
    return f


def cv_pred(X, y, w, k=10, reps=3, groups=None):
    """Out-of-fold predictions of y from the columns of X (average over reps; seeds 100+r)."""
    P = np.zeros(len(y))
    for r in range(reps):
        f = folds(y, k, 100 + r, groups)
        for j in range(k):
            tr, te = f != j, f == j
            if not te.any():
                P[te] += 0
                continue
            P[te] += pred_lr(fit_lr(X[tr], y[tr], w[tr]), X[te]) / reps
    return P


def ce_bits(p, y, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(w * (y * np.log2(p) + (1 - y) * np.log2(1 - p))).sum() / w.sum())


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cv_gain_pct(s, y, w, groups=None, k=10, reps=3):
    """Cross-validated gain: H(Y) minus the out-of-fold cross-entropy of a one-column logistic
    regression on logit(s), in % of H(Y). Positive = usable information out of sample."""
    HY = h2((w * y).sum() / w.sum())
    pc = cv_pred(logit(s)[:, None], y, w, k=k, reps=reps, groups=groups)
    return 100 * (HY - ce_bits(pc, y, w)) / HY


def mi_pct(s, y, w, nperm=500, seed=5):
    """5-bin mutual information minus the mean of a permutation null, in % of H(Y), and its p."""
    HY = h2((w * y).sum() / w.sum())
    b = wbins(s, w)
    I = wmi(b, y, w)
    rng = np.random.default_rng(seed)
    null = np.array([wmi(b[rng.permutation(len(b))], y, w) for _ in range(nperm)])
    return 100 * (I - null.mean()) / HY, float(((null >= I).sum() + 1) / (nperm + 1)), b


def threshold_for_best(s, w, value):
    """Among skipping the top 0..100% (5% steps), the threshold with the best value per row
    (skip when s >= threshold). Skipping everything is -inf."""
    o = np.argsort(-s); cw = np.cumsum(w[o]) / w.sum()
    best_v, best_t = None, np.inf
    for fr in np.arange(0, 1.0001, 0.05):
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


def profit_xfit(s, y, value, w, half):
    """Pick the best skip share (5% steps) on half 1 and apply its threshold to half 2, and the reverse."""
    tot = kept = kneg = val = 0.0
    for a, b in ((1, 2), (2, 1)):
        ma, mb = half == a, half == b
        if not ma.any() or not mb.any():
            return None
        thr = threshold_for_best(s[ma], w[ma], value[ma])
        keep = s[mb] < thr
        wb = w[mb]
        tot += wb.sum(); kept += wb[keep].sum(); kneg += (wb[keep] * (1 - y[mb][keep])).sum(); val += (wb[keep] * value[mb][keep]).sum()
    return {"value_per_1000": round(1000 * val / tot, 1), "keep_frac": round(kept / tot, 3),
            "kept_neg_pct": round(100 * kneg / kept, 2) if kept > 0 else None}


def metrics(s, y, w, value, half, groups=None, nperm=500, k=10, reps=3):
    """The standard table for one prompt on one split."""
    y = np.asarray(y, float); w = np.asarray(w, float); s = np.asarray(s, float); value = np.asarray(value, float)
    half = np.asarray(half)
    mi, mi_p, b = mi_pct(s, y, w, nperm=nperm)
    gain = cv_gain_pct(s, y, w, groups=groups, k=k, reps=reps)
    pos_rate = []
    for q in range(5):
        m = b == q
        pos_rate.append(round(100 * float((w[m] * y[m]).sum() / w[m].sum()), 2) if m.any() else None)
    out = {"auc": _r(wauc(s, y, w), 4),
           "auc_h1": _r(wauc(s[half == 1], y[half == 1], w[half == 1]), 4) if (half == 1).any() else None,
           "auc_h2": _r(wauc(s[half == 2], y[half == 2], w[half == 2]), 4) if (half == 2).any() else None,
           "mi_pct": round(mi, 3), "mi_p": round(mi_p, 4), "cv_gain_pct": round(gain, 3),
           "pos_by_quintile_pct": pos_rate}
    if (half > 0).any():
        out["xfit"] = profit_xfit(s, y, value, w, half)
    return out


def decision(s, y, w, value, thr=0.5):
    """Use the answer as the rule says: skip when s >= thr."""
    y = np.asarray(y, float); w = np.asarray(w, float); s = np.asarray(s, float); value = np.asarray(value, float)
    kp = s < thr
    W = w.sum()
    kept_pos = (w[kp] * y[kp]).sum() / w[kp].sum() if kp.any() else None
    kept_neg = (w[kp] * (1 - y[kp])).sum() / w[kp].sum() if kp.any() else None
    skip_pos = (w[~kp] * y[~kp]).sum() / w[~kp].sum() if (~kp).any() else None
    v = 1000 * (w[kp] * value[kp]).sum() / W
    allin = 1000 * (w * value).sum() / W
    return {"threshold": thr,
            "skip_frac": round(float(w[~kp].sum() / W), 3),
            "kept_pos_pct": _r(100 * kept_pos, 2) if kept_pos is not None else None,
            "kept_neg_pct": _r(100 * kept_neg, 2) if kept_neg is not None else None,
            "skipped_pos_pct": _r(100 * skip_pos, 2) if skip_pos is not None else None,
            "lift": _r(skip_pos / kept_pos, 3) if (kept_pos and skip_pos is not None) else None,
            "value_per_1000": round(float(v), 1),
            "all_in_value_per_1000": round(float(allin), 1),
            "gain_vs_all_in_per_1000": round(float(v - allin), 1)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - r) / d), float((c + r) / d))


def final_summary(s, y, value, thr, nperm=2000, seed=9):
    """The one-shot test table: rows taken, rates with Wilson 95% intervals, value per row and in
    total, AUC with a one-sided permutation p, and MI corrected by its permutation null."""
    s = np.asarray(s, float); y = np.asarray(y, float); value = np.asarray(value, float)
    w = np.ones(len(y))
    keep = s < thr
    n = int(keep.sum()); kpos = int((keep & (y == 1)).sum())
    lo, hi = wilson(n - kpos, n)
    plo, phi = wilson(kpos, n)
    rng = np.random.default_rng(seed)
    out = {"rows": int(len(y)), "taken": n, "skipped": int(len(y) - n), "skip_frac": round(1 - n / len(y), 3) if len(y) else None,
           "taken_pos": kpos, "taken_neg": n - kpos,
           "taken_neg_pct": round(100 * (n - kpos) / n, 2) if n else None,
           "taken_neg_ci95": [round(100 * lo, 1), round(100 * hi, 1)] if n else None,
           "taken_pos_pct": round(100 * kpos / n, 2) if n else None,
           "taken_pos_ci95": [round(100 * plo, 1), round(100 * phi, 1)] if n else None,
           "value_per_taken": round(float(value[keep].mean()), 3) if n else None,
           "value_total": round(float(value[keep].sum()), 1)}
    if len(np.unique(s)) > 1 and 0 < y.sum() < len(y):
        a = wauc(s, y, w)
        pa = (sum(wauc(s, rng.permutation(y), w) >= a for _ in range(nperm)) + 1) / (nperm + 1)
        HY = h2(y.mean())
        b = wbins(s, w); I = wmi(b, y, w)
        null = np.array([wmi(b[rng.permutation(len(b))], y, w) for _ in range(nperm)])
        out.update({"auc": round(a, 4), "auc_p_one_sided": round(float(pa), 4),
                    "mi_pct": round(100 * (I - null.mean()) / HY, 3), "mi_p": round(float(((null >= I).sum() + 1) / (nperm + 1)), 4)})
    return out


def percentile(null_values, x):
    """Where x falls in the null distribution, in percent (ties count half)."""
    v = np.asarray([z for z in null_values if z is not None and np.isfinite(z)], float)
    if x is None or not len(v) or not np.isfinite(x):
        return None
    return round(100 * (float((v < x).sum()) + 0.5 * float((v == x).sum())) / len(v), 1)


def _r(x, k):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), k)
