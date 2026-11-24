"""promptlab.yardsticks against version 0.1.0 and against the theory.

A two-class outcome with one question must give the numbers of 0.1.0 float for float. The oracle is
tests/golden/v01_metrics.py, run in this process. Three-class cases are held to values worked out
from the theory in README.ja.md, never to a run of the code under test.
"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from promptlab import yardsticks as Y
from tests.golden import v01_metrics as V


def two_class_units(seed, split, n=2400):
    """Units like one split of a study, with the kinds of answer a yes/no question gets.

    "train": every positive unit, then a seeded pick of six negatives per positive that carry the
    weight of the ones left out, in two halves. "any weights": every unit with a weight of its own,
    in two halves. Any other split: every unit, weight 1, no halves.
    """
    rng = np.random.default_rng(seed)
    month = np.sort(rng.integers(0, 48, n))
    hit = rng.random(n) < 0.2
    y = (rng.random(n) < np.where(hit, 0.085, 0.05)).astype(float)
    if split == "train":
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        k = min(len(neg), 6 * len(pos))
        idx = np.r_[pos, neg[np.sort(np.random.default_rng(12345).choice(len(neg), k, replace=False))]]
        w = np.r_[np.ones(len(pos)), np.full(k, len(neg) / k)]
        half = np.where(month[idx] < 24, 1, 2)
    elif split == "any weights":
        idx, w, half = np.arange(n), rng.uniform(0.5, 3, n), np.where(month < 24, 1, 2)
    else:
        idx, w, half = np.arange(n), np.ones(n), np.zeros(n, int)
    y, hit, m = y[idx], hit[idx], len(idx)
    groups = np.array([f"{2018 + q // 12}-{q % 12 + 1:02d}" for q in month[idx]])
    value = np.where(y == 1, -50.0, 2.0)
    answers = {
        "machine": hit.astype(float),
        "fake": np.round(np.where(hit, 0.94, 0.06) + (rng.random(m) - 0.5) * 0.08, 2),
        "uniform": rng.random(m),
        "below the clip": rng.random(m) ** 8,
        "refused": np.full(m, 0.5),
        "score mean": np.round(rng.random(m) * 4, 1) / 4,
        "the outcome itself": y.copy(),
    }
    labels = Y.Labels(z=y.astype(int), K=2, w=w, groups=groups, half=half, value=value)
    return SimpleNamespace(y=y, w=w, groups=groups, half=half, value=value, labels=labels, answers=answers)


def plain_labels(z):
    """Two-class units that all weigh 1 and share one group, with no halves and no value."""
    z = np.asarray(z)
    return Y.Labels(z=z, K=2, w=np.ones(len(z)), groups=np.full(len(z), "g"), half=np.zeros(len(z), int), value=None)


TRAIN = {seed: two_class_units(seed, "train") for seed in (1, 2, 3, 7)}
EVAL = {5: two_class_units(5, "eval", n=1200)}
ANY_WEIGHTS = {6: two_class_units(6, "any weights", n=600)}
TEST = two_class_units(9, "test", n=600)


def cases(units):
    for seed, u in units.items():
        for name, s in u.answers.items():
            yield {"seed": seed, "answers": name}, u, s


class TwoClassesOneQuestion(unittest.TestCase):
    """Float for float the numbers of 0.1.0."""

    def test_entropy(self):
        for seed, u in TRAIN.items():
            self.assertEqual(Y.entropy_bits(u.labels), V.h2((u.w * u.y).sum() / u.w.sum()), seed)
        for only in (0, 1):
            self.assertEqual(Y.entropy_bits(plain_labels([only] * 8)), V.h2(float(only)))

    def test_a_yes_no_answer_encodes_to_its_logit(self):
        edges = np.array([0.0, 1e-9, 1e-5, 1e-4, 0.5, 1 - 1e-4, 1 - 1e-5, 1 - 1e-9, 1.0])
        for s in (edges, *(s for _, _, s in cases(TRAIN))):
            np.testing.assert_array_equal(Y.log_ratios(s[:, None])[:, 0], V.logit(s))

    def test_recalibrated_gain(self):
        for what, u, s in cases(TRAIN):
            with self.subTest(**what):
                self.assertEqual(Y.recalibrated_gain_pct(Y.log_ratios(s[:, None]), u.labels, 10, 3),
                                 V.cv_gain_pct(s, u.y, u.w, groups=u.groups, k=10, reps=3))

    def test_cross_entropy_at_the_clip(self):
        p = np.array([1e-9, 1e-6, 0.5, 1 - 1e-6, 1 - 1e-9, 1.0, 0.0])
        z = np.array([1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(Y.ce_bits(p[:, None], plain_labels(z)), V.ce_bits(p, z.astype(float), np.ones(7)))

    def test_two_columns_fit_predict_and_cross_validate(self):
        for seed, u in TRAIN.items():
            with self.subTest(seed=seed):
                X = np.column_stack([V.logit(u.answers["uniform"]), V.logit(u.answers["fake"])])
                b = V.fit_lr(X, u.y, u.w)
                B = Y.fit(X, u.labels.z, u.w, 2)
                np.testing.assert_array_equal(B, b[None, :])
                np.testing.assert_array_equal(Y.predict_rest(B, X)[:, 0], V.pred_lr(b, X))
                np.testing.assert_array_equal(Y.cv_predict(X, u.labels, 10, 3)[:, 0],
                                              V.cv_pred(X, u.y, u.w, k=10, reps=3, groups=u.groups))

    def test_fit_where_the_clip_and_the_iteration_limit_bind(self):
        u = TRAIN[1]
        X = ((2 * u.y - 1) * 1e4)[:, None]
        b = V.fit_lr(X, u.y, u.w)
        B = Y.fit(X, u.labels.z, u.w, 2)
        np.testing.assert_array_equal(B, b[None, :])
        np.testing.assert_array_equal(Y.predict_rest(B, X)[:, 0], V.pred_lr(b, X))

    def test_auc_whole_and_by_half(self):
        for what, u, s in cases(TRAIN):
            for m in (u.half > 0, u.half == 1, u.half == 2):
                self.assertEqual(Y.wauc(s[m], u.y[m], u.w[m]), V.wauc(s[m], u.y[m], u.w[m]), what)

    def test_binned_mi(self):
        for what, u, s in cases({1: TRAIN[1]}):
            with self.subTest(**what):
                mi, p, bins = Y.binned_mi_pct(s, u.labels, 500)
                old_mi, old_p, old_bins = V.mi_pct(s, u.y, u.w, nperm=500)
                self.assertEqual((mi, p), (old_mi, old_p))
                np.testing.assert_array_equal(bins, old_bins)

    def test_record_block(self):
        for what, u, s in cases({1: TRAIN[1], **EVAL, **ANY_WEIGHTS}):
            with self.subTest(**what):
                old = V.metrics(s, u.y, u.w, u.value, u.half, groups=u.groups, nperm=50, k=10, reps=3)
                old["decision"] = V.decision(s, u.y, u.w, u.value, 0.5)
                h = V.h2((u.w * u.y).sum() / u.w.sum())
                block = Y.score(Y.log_ratios(s[:, None]), u.labels, cv_folds=10, cv_reps=3, mi_perms=50,
                                forecast=Y.Forecast(Y.forecast_rest(s[:, None], (0, 1), 2), fitted=False), threshold=0.5)
                self.assertEqual(block.pop("asis_gain_pct"), round(100 * (h - V.ce_bits(s, u.y, u.w)) / h, 3))
                self.assertEqual(block, old)
                self.assertEqual(list(block), list(old))

    def test_which_keys_the_record_block_holds(self):
        u = TRAIN[1]
        s = u.answers["fake"]
        X, settings = Y.log_ratios(s[:, None]), {"cv_folds": 5, "cv_reps": 1, "mi_perms": 10}
        own, fitted = Y.Forecast(s[:, None], fitted=False), Y.Forecast(s[:, None], fitted=True)
        scalar = ["auc", "auc_h1", "auc_h2", "mi_pct", "mi_p", "cv_gain_pct"]
        self.assertEqual(list(Y.score(X, u.labels, **settings, forecast=fitted)), [*scalar, "pos_by_quintile_pct", "xfit"])
        self.assertEqual(list(Y.score(X, replace(u.labels, value=None), **settings, forecast=own, threshold=0.5)),
                         [*scalar, "asis_gain_pct", "pos_by_quintile_pct"])
        skip_everything = Y.score(X, u.labels, **settings, forecast=own, threshold=0.0)
        self.assertEqual(skip_everything["decision"], V.decision(s, u.y, u.w, u.value, 0.0))

    def test_forecast_of_a_yes_no_question(self):
        p = TRAIN[1].answers["uniform"][:, None]
        np.testing.assert_array_equal(Y.forecast_rest(p, (0, 1), 2), p)
        np.testing.assert_array_equal(Y.forecast_rest(p, (1, 0), 2), 1 - p)

    def test_threshold_for_best_when_a_larger_skip_is_worth_the_same(self):
        s, w = np.arange(20, 0, -1.0), np.ones(20)
        value = np.r_[-50.0, 0.0, np.full(18, 2.0)]
        self.assertEqual(Y.threshold_for_best(s, w, value), V.threshold_for_best(s, w, value))

    def test_final_summary(self):
        for s, threshold in ((TEST.answers["fake"], 0.5), (TEST.answers["refused"], 0.5), (np.zeros(len(TEST.y)), np.inf)):
            self.assertEqual(Y.final_summary(s, TEST.labels, threshold), V.final_summary(s, TEST.y, TEST.value, threshold))

    def test_final_summary_counts_every_unit_once(self):
        weighted = two_class_units(4, "train", n=900)
        s = weighted.answers["uniform"]
        self.assertEqual(Y.final_summary(s, weighted.labels, 0.3), V.final_summary(s, weighted.y, weighted.value, 0.3))

    def test_percentile(self):
        for draws, x in (([1, 2, 3, 4], 3), ([1, 2, 3, 4], 10), ([], 1), ([1.0, None, float("nan"), 2.0], 1.5),
                         ([1, 2], None), ([1, 2], float("inf"))):
            self.assertEqual(Y.percentile(draws, x), V.percentile(draws, x))


H3 = math.log2(3)
CHANNEL_PCT = 100 * (H3 + 0.8 * math.log2(0.8) + 0.2 * math.log2(0.1)) / H3


def three_class_channel(seed=11, n=3000):
    """A three-class outcome and the option a decision model leans to: the true class eight times
    in ten, otherwise one of the other two. Knowing the lean is worth CHANNEL_PCT of H(Z)."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 3, n)
    lean = np.where(rng.random(n) < 0.8, z, (z + rng.integers(1, 3, n)) % 3)
    groups = np.array([f"g{v:02d}" for v in rng.integers(0, 40, n)])
    return z, lean, Y.Labels(z=z, K=3, w=np.ones(n), groups=groups, half=np.zeros(n, int), value=None)


def choice_answers(lean, top):
    """A three-option choice that puts `top` on the leaned-to option and splits the remainder."""
    probs = np.full((len(lean), 3), (1 - top) / 2)
    probs[np.arange(len(lean)), lean] = top
    return probs[:, 1:]


def gain_from_counts(z, lean, top):
    """100 (H - CE) / H of choice_answers(lean, top), from the sample counts alone."""
    h = -sum(p * math.log2(p) for p in np.bincount(z, minlength=3) / len(z))
    right = (lean == z).mean()
    return 100 * (h + right * math.log2(top) + (1 - right) * math.log2((1 - top) / 2)) / h


def constant_forecast_gain(labels, k, reps):
    """What V scores when the answers say nothing: each fold is forecast by the class shares of the
    other folds, with the folds of 0.1.0."""
    z, n = labels.z, len(labels.z)
    P = np.zeros((n, labels.K))
    for r in range(reps):
        f = V.folds(None, k, 100 + r, labels.groups)
        for j in range(k):
            P[f == j] += np.bincount(z[f != j], minlength=labels.K) / (f != j).sum() / reps
    h = -sum(p * math.log2(p) for p in np.bincount(z, minlength=labels.K) / n)
    return 100 * (h + np.log2(P[np.arange(n), z]).mean()) / h


class ThreeClasses(unittest.TestCase):
    """The behaviour the theory in README.ja.md predicts for an outcome with more than two classes."""

    @classmethod
    def setUpClass(cls):
        cls.z, cls.lean, cls.labels = three_class_channel()

    def gains(self, rest):
        asis = Y.gain_pct(Y.forecast_rest(rest, (0, 1, 2), 3), self.labels)
        return asis, Y.recalibrated_gain_pct(Y.log_ratios(rest), self.labels, 10, 3)

    def test_calibrated_answers_recover_the_information_of_the_channel(self):
        asis, cv = self.gains(choice_answers(self.lean, 0.8))
        self.assertAlmostEqual(asis, gain_from_counts(self.z, self.lean, 0.8), places=6)
        self.assertLess(abs(asis - CHANNEL_PCT), 4)
        self.assertLess(abs(cv - asis), 0.5)

    def test_overconfidence_costs_the_as_is_gain_and_recalibration_recovers_it(self):
        asis, cv = self.gains(choice_answers(self.lean, 0.998))
        self.assertAlmostEqual(asis, gain_from_counts(self.z, self.lean, 0.998), places=6)
        self.assertLess(asis, 0)
        self.assertAlmostEqual(cv, self.gains(choice_answers(self.lean, 0.8))[1], delta=0.05)

    def test_noise_scores_slightly_below_zero(self):
        rng = np.random.default_rng(12)
        noise = [Y.recalibrated_gain_pct(Y.log_ratios(rng.dirichlet(np.ones(3), len(self.z))[:, 1:]), self.labels, 10, 3)
                 for _ in range(6)]
        self.assertLess(np.mean(noise), 0)
        self.assertGreater(np.mean(noise), -1)

    def test_a_uniform_answer_carries_no_information(self):
        n = len(self.z)
        for uniform in (np.full((n, 1), 1 / 2), np.full((n, 2), 1 / 3), np.full((n, 3), 1 / 4)):
            np.testing.assert_allclose(Y.log_ratios(uniform), 0, atol=1e-12)
        said_nothing = Y.log_ratios(np.full((n, 2), 1 / 3))
        self.assertAlmostEqual(Y.recalibrated_gain_pct(said_nothing, self.labels, 10, 3),
                               constant_forecast_gain(self.labels, 10, 3), places=4)
        informative = Y.log_ratios(choice_answers(self.lean, 0.8))
        self.assertAlmostEqual(Y.recalibrated_gain_pct(np.column_stack([informative, said_nothing]), self.labels, 10, 3),
                               Y.recalibrated_gain_pct(informative, self.labels, 10, 3), places=6)

    def test_the_fit_is_the_best_member_of_the_family(self):
        rng = np.random.default_rng(3)
        n = len(self.z)
        X = rng.normal(0, 1, (n, 2)) + 1.2 * (self.z[:, None] == [1, 2])
        w = rng.uniform(0.5, 3, n)
        X1 = np.column_stack([np.ones(n), X])

        def penalized_log_loss(B):
            eta = np.column_stack([np.zeros(n), X1 @ B.T])
            log_p = eta - np.log(np.exp(eta).sum(axis=1, keepdims=True))
            return -(w * log_p[np.arange(n), self.z]).sum() + 0.5 * 1e-3 * (B[:, 1:] ** 2).sum()

        B = Y.fit(X, self.z, w, 3)
        odds = np.exp(X1 @ B.T)
        np.testing.assert_allclose(Y.predict_rest(B, X), odds / (1 + odds.sum(axis=1, keepdims=True)))
        for _ in range(50):
            self.assertGreater(penalized_log_loss(B + rng.normal(0, 1e-3, B.shape)), penalized_log_loss(B))

    def test_record_block_keeps_to_the_information_yardsticks(self):
        rest = choice_answers(self.lean, 0.8)
        settings = {"cv_folds": 5, "cv_reps": 1, "mi_perms": 10}
        forecast = Y.Forecast(Y.forecast_rest(rest, (0, 1, 2), 3), fitted=False)
        self.assertEqual(list(Y.score(Y.log_ratios(rest), self.labels, **settings)), ["cv_gain_pct"])
        self.assertEqual(list(Y.score(Y.log_ratios(rest), self.labels, **settings, forecast=forecast, threshold=0.5)),
                         ["cv_gain_pct", "asis_gain_pct"])


class AsIsGain(unittest.TestCase):
    """The worked examples of the theory in README.ja.md: statements true half the time, so H(Z) is one bit."""

    def bits(self, said_true):
        return Y.gain_pct(np.array(said_true)[:, None], plain_labels([1, 1, 0, 0])) / 100

    def test_one_confident_miss_in_four_costs_more_than_the_three_hits_earn(self):
        self.assertAlmostEqual(self.bits([0.99, 0.99, 0.01, 0.99]), -0.672, places=3)

    def test_nothing_read_summary_full_text(self):
        for said_true, bits in (([0.5, 0.5, 0.5, 0.5], 0.000), ([0.8, 0.6, 0.2, 0.4], 0.471), ([0.95, 0.9, 0.05, 0.1], 0.887)):
            self.assertAlmostEqual(self.bits(said_true), bits, places=3)


class Sign(unittest.TestCase):
    """the theory in README.ja.md section 3: the recalibrated gain does not see which way an answer points; the
    AUC and the as-is gain do."""

    def test_reversing_the_answer(self):
        u = TRAIN[1]
        s = u.answers["fake"]
        self.assertAlmostEqual(Y.recalibrated_gain_pct(Y.log_ratios((1 - s)[:, None]), u.labels, 10, 3),
                               Y.recalibrated_gain_pct(Y.log_ratios(s[:, None]), u.labels, 10, 3), places=6)
        self.assertAlmostEqual(Y.wauc(1 - s, u.y, u.w), 1 - Y.wauc(s, u.y, u.w), places=9)
        self.assertLess(Y.gain_pct((1 - s)[:, None], u.labels), Y.gain_pct(s[:, None], u.labels))


class Encoding(unittest.TestCase):
    def test_a_choice_encodes_each_option_against_the_first(self):
        np.testing.assert_allclose(Y.log_ratios(np.array([[0.5, 0.3]])), [[math.log(0.5 / 0.2), math.log(0.3 / 0.2)]])

    def test_a_choice_named_without_probabilities_encodes_as_a_point_mass_at_the_clip(self):
        top = math.log((1 - V.EPS) / V.EPS)
        named_first, named_second, named_third = Y.log_ratios(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_allclose(named_first, [-top, -top], atol=1e-3)
        np.testing.assert_allclose(named_second, [top, 0], atol=1e-3)
        np.testing.assert_allclose(named_third, [0, top], atol=1e-3)

    def test_forecast_adds_the_options_of_each_class(self):
        rest = np.array([[0.1, 0.2, 0.3]])
        np.testing.assert_allclose(Y.forecast_rest(rest, (0, 1, 2, 1), 3), [[0.4, 0.2]])
        np.testing.assert_allclose(Y.forecast_rest(rest, (2, 0, 0, 1), 3), [[0.3, 0.4]])


class FittedForecastAUC(unittest.TestCase):
    """The AUC of an out-of-fold forecast, pooled over folds, falls below 0.5 for a weak answer;
    taken within the folds it does not."""

    def units(self, seed, effect):
        rng = np.random.default_rng(seed)
        n = 1500
        month = rng.integers(0, 24, n)
        monthly_logit = -2.6 + rng.normal(0, 0.35, 24)[month]
        a = (rng.random(n) < 0.15).astype(float)
        z = (rng.random(n) < 1 / (1 + np.exp(-(monthly_logit + effect * a)))).astype(int)
        labels = Y.Labels(z=z, K=2, w=np.ones(n), groups=month, half=np.where(month < 12, 1, 2), value=None)
        return a, Y.log_ratios(a[:, None]), labels

    def test_within_the_folds(self):
        for effect in (0.45, 0.0):
            found = []
            for seed in range(12):
                a, X, L = self.units(seed, effect)
                found.append((Y.wauc(a, L.y, L.w), Y.wauc(Y.cv_predict(X, L, 10, 3)[:, 0], L.y, L.w), Y.cv_auc(X, L, 10, 3)))
            answer, pooled = np.mean([x[0] for x in found]), np.mean([x[1] for x in found])
            within = np.mean([x[2][0] for x in found])
            self.assertLess(pooled, answer - 0.05)
            self.assertAlmostEqual(within, answer, delta=0.02)
            self.assertTrue(all(x[2][1] is not None and x[2][2] is not None for x in found))

    def test_a_part_without_a_pair_has_none(self):
        a, X, L = self.units(0, 0.45)
        self.assertEqual(Y.cv_auc(X, replace(L, half=np.zeros(len(a), int)), 10, 1)[1:], (None, None))


def softmax_units(seed, intercepts, slopes, n=3000, columns=None):
    """Units whose class k > 0 has the score intercepts[k-1] + slopes[k-1] . x over the answer
    columns x, (n, d). Without `columns`, one standard normal column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1)) if columns is None else columns(rng, n)
    eta = np.column_stack([a + X @ np.atleast_1d(b) for a, b in zip(intercepts, slopes)])
    P = np.exp(eta) / (1 + np.exp(eta).sum(axis=1, keepdims=True))
    u = rng.random(n)[:, None]
    z = (u > np.cumsum(np.column_stack([1 - P.sum(axis=1), P]), axis=1)).sum(axis=1)
    labels = Y.Labels(z=z, K=len(intercepts) + 1, w=np.ones(n), groups=rng.integers(0, 12, n), half=np.zeros(n, int), value=None)
    return X, labels


class Transfer(unittest.TestCase):
    """transfer_gain_pct keeps the direction V was fitted to on the training split; cv_gain_pct refits it."""

    def gains(self, train, ev):
        (Xt, Lt), (Xe, Le) = train, ev
        B = Y.fit(Xt, Lt.z, Lt.w, Lt.K)
        return Y.recalibrated_gain_pct(Xe, Le, 10, 3), Y.transfer_gain_pct(Xe, B, Le, 10, 3)

    def test_a_relation_that_turned_around_scores_at_most_zero_while_cv_gain_stays_up(self):
        cv, transfer = self.gains(softmax_units(1, [-2.7], [0.8]), softmax_units(2, [-2.7], [-0.8]))
        self.assertGreater(cv, 3)
        self.assertLessEqual(transfer, 0)

    def test_a_relation_that_held_scores_as_cv_gain_though_the_class_shares_moved(self):
        cv, transfer = self.gains(softmax_units(1, [-2.7], [0.8]), softmax_units(3, [-3.4], [0.8]))
        self.assertGreater(transfer, 3)
        self.assertAlmostEqual(transfer, cv, delta=0.3)

    def test_a_weaker_relation_keeps_a_positive_gain(self):
        self.assertGreater(self.gains(softmax_units(1, [-2.7], [0.8]), softmax_units(4, [-2.7], [0.4]))[1], 0.5)

    def test_a_stable_topic_mixed_with_one_that_turned_around(self):
        """The shape of the synthetic ticket task: cv_gain prefers the answers that read both topics,
        transfer_gain the answers that read the stable one alone."""
        def topics(rng, n):
            return np.column_stack([Y.log_ratios((rng.random(n) < p).astype(float)[:, None]) for p in (0.18, 0.15)])
        slope_per_logit = 1 / (2 * math.log((1 - Y.LOGIT_EPS) / Y.LOGIT_EPS))
        train = softmax_units(100, [-2.6], [np.array([0.95, 0.7]) * slope_per_logit], columns=topics)
        ev = softmax_units(101, [-2.6], [np.array([0.95, -0.9]) * slope_per_logit], columns=topics)
        both, stable = self.gains(train, ev), self.gains((train[0][:, :1], train[1]), (ev[0][:, :1], ev[1]))
        self.assertGreater(both[0], stable[0])
        self.assertLess(both[1], stable[1])

    def test_three_classes(self):
        cv, transfer = self.gains(softmax_units(7, [-1.0, -1.5], [0.8, -0.6]), softmax_units(8, [-1.0, -1.5], [-0.8, 0.6]))
        self.assertGreater(cv, 3)
        self.assertLessEqual(transfer, 0)
        cv, transfer = self.gains(softmax_units(7, [-1.0, -1.5], [0.8, -0.6]), softmax_units(9, [-1.2, -1.2], [0.8, -0.6]))
        self.assertAlmostEqual(transfer, cv, delta=0.3)

    def test_the_fit_is_the_best_member_with_a_strength_of_at_least_zero(self):
        for slopes, turned_around in (([0.8, -0.6], False), ([-0.8, 0.6], True)):
            X, L = softmax_units(10, [-1.0, -1.5], slopes)
            E = X @ np.array([[0.8, -0.6]])
            a, c = Y.fit_along(E, L.z, L.w, L.K)

            def nll(a, c):
                return Y.ce_bits(Y._along(a, c, E), L) + Y.RIDGE * c ** 2 / 2 / np.log(2) / L.w.sum()

            best = nll(a, c)
            for da in ([0.01, 0], [0, 0.01], [-0.01, 0], [0, -0.01]):
                self.assertLessEqual(best, nll(a + np.array(da), c) + 1e-12)
            for dc in (0.01, -0.01):
                if c + dc >= 0:
                    self.assertLessEqual(best, nll(a, c + dc) + 1e-12)
            self.assertEqual(c == 0.0, turned_around)

    def test_a_record_of_another_split_holds_it(self):
        X, L = softmax_units(11, [-2.7], [0.8])
        block = Y.score(X, L, cv_folds=10, cv_reps=3, mi_perms=10, trained=Y.fit(X, L.z, L.w, 2))
        self.assertEqual(list(block), ["cv_gain_pct", "transfer_gain_pct"])
        self.assertNotIn("transfer_gain_pct", Y.score(X, L, cv_folds=10, cv_reps=3, mi_perms=10))


if __name__ == "__main__":
    unittest.main()
