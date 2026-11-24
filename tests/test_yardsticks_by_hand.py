"""Yardsticks checked against values worked out by hand or by a slower, plainer computation."""
import itertools
import unittest

import numpy as np

from promptlab import yardsticks as Y


def labels(y, value=None, w=None, half=None):
    y = np.asarray(y)
    n = len(y)
    return Y.Labels(z=y.astype(int), K=2, w=np.ones(n) if w is None else w, groups=np.full(n, "g"),
                    half=np.zeros(n, int) if half is None else half, value=value)


class TestRanking(unittest.TestCase):
    def test_wauc_is_the_weighted_share_of_pairs_ranked_right(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            n = 60
            s = rng.integers(0, 6, n).astype(float)
            y = (rng.random(n) < 0.3).astype(float)
            w = rng.uniform(0.5, 2, n)
            right = pairs = 0.0
            for i, j in itertools.product(range(n), range(n)):
                if y[i] == 1 and y[j] == 0:
                    pairs += w[i] * w[j]
                    right += w[i] * w[j] * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
            self.assertAlmostEqual(Y.wauc(s, y, w), right / pairs, places=10)

    def test_percentile_counts_ties_as_half(self):
        self.assertEqual(Y.percentile([1, 2, 3, 4], 3), 62.5)
        self.assertEqual(Y.percentile([1, 2, 3, 4], 10), 100.0)
        self.assertIsNone(Y.percentile([], 1))


class TestValue(unittest.TestCase):
    def test_decision_by_hand(self):
        s = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        y = np.array([1, 0, 1, 0, 0, 0])
        d = Y.decision(s, labels(y, value=np.where(y == 1, -50.0, 2.0)), 0.5)
        self.assertEqual(d["skip_frac"], round(2 / 6, 3))
        self.assertEqual((d["kept_pos_pct"], d["kept_neg_pct"], d["skipped_pos_pct"], d["lift"]), (25.0, 75.0, 50.0, 2.0))
        self.assertEqual(d["value_per_1000"], round(1000 * (-50 + 2 + 2 + 2) / 6, 1))
        self.assertEqual(d["all_in_value_per_1000"], round(1000 * (-100 + 8) / 6, 1))
        self.assertAlmostEqual(d["gain_vs_all_in_per_1000"], d["value_per_1000"] - d["all_in_value_per_1000"], places=1)

    def test_the_best_threshold_skips_the_two_losers(self):
        s = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        value = np.array([-50, -50] + [2] * 18, float)
        self.assertEqual(Y.threshold_for_best(s, np.ones(20), value), 9)

    def test_the_cross_fit_needs_both_halves(self):
        one_half = labels(np.zeros(4), value=np.ones(4), half=np.ones(4, int))
        self.assertIsNone(Y.profit_xfit(np.zeros(4), one_half))

    def test_rule_table_by_hand(self):
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0])
        hit = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)
        w = np.array([1, 1, 1, 1, 2, 2, 2, 2], float)
        table = Y.rule_table([hit, None, np.zeros(8, bool)], labels(y, w=w))
        self.assertEqual(table[0], {"rule": 1, "hit_frac": round(2 / 12, 3), "pos_pct_hit": 50.0, "pos_pct_rest": 10.0, "lift": 5.0})
        self.assertEqual(table[1], {"rule": 2, "expr": None})
        self.assertEqual(table[2], {"rule": 3, "hit_frac": 0.0, "pos_pct_hit": None, "pos_pct_rest": round(100 * 2 / 12, 2), "lift": None})

    def test_the_test_table_counts_every_unit_once(self):
        rng = np.random.default_rng(5)
        y = (rng.random(400) < 0.1).astype(int)
        s = np.clip(0.5 * y + 0.3 * rng.random(400), 0, 1)
        lab = labels(y, value=np.where(y == 1, -50.0, 2.0), w=rng.uniform(1, 9, 400))
        table = Y.final_summary(s, lab, 0.5)
        self.assertEqual(table["taken"], int((s < 0.5).sum()))
        self.assertEqual(table["taken"] + table["skipped"], 400)
        self.assertEqual(table["value_total"], round(float(lab.value[s < 0.5].sum()), 1))
        self.assertLess(table["auc_p_one_sided"], 0.05)
        everything = Y.final_summary(np.zeros(400), lab, np.inf)
        self.assertEqual(everything["taken"], 400)
        self.assertNotIn("auc", everything)


if __name__ == "__main__":
    unittest.main()
