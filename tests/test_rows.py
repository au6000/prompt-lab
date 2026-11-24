import json
import unittest
from datetime import datetime, timezone

from tests.helpers import TempTask, alert_rows, alert_task


def when(row):
    return datetime.strptime(row["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask(rows=alert_rows(800))

    def tearDown(self):
        self.tt.cleanup()

    def test_rows_go_to_their_period_in_time_order(self):
        out = self.tt.ok("split")
        bounds = {"train": (2018, 2022), "eval": (2022, 2024), "test": (2024, 2026)}
        total = 0
        for split, (first, last) in bounds.items():
            rows = self.tt.split_rows(split)
            total += len(rows)
            self.assertTrue(all(first <= when(r).year < last for r in rows))
            self.assertEqual([when(r) for r in rows], sorted(when(r) for r in rows))
        self.assertEqual(total, 800)
        self.assertIn("dropped (outside every range): 0", out)

    def test_the_summary_shows_training_outcomes_only(self):
        lines = self.tt.ok("split").splitlines()
        self.assertIn("classes", lines[0])
        self.assertRegex(lines[1], r"^eval : \d+ rows$")
        self.assertRegex(lines[2], r"^test : \d+ rows$")

    def test_a_split_is_not_redone_silently(self):
        self.tt.ok("split")
        self.assertEqual(self.tt.plab("split").code, 2)
        self.tt.ok("split", "--force")

    def test_periods_are_half_open(self):
        self.tt.write_rows([{"id": "a", "t": "2021-12-31T23:59:59Z", "label": 0, "value": 1, "state": {}},
                            {"id": "b", "t": "2022-01-01T00:00:00Z", "label": 1, "value": -1, "state": {}},
                            {"id": "c", "t": "2022-01-01 00:00:00", "label": 0, "value": 1, "state": {}},
                            {"id": "d", "t": "2030-01-01", "label": 0, "value": 1, "state": {}}])
        out = self.tt.ok("split")
        self.assertEqual([r["id"] for r in self.tt.split_rows("train")], ["a"])
        self.assertEqual([r["id"] for r in self.tt.split_rows("eval")], ["b", "c"])
        self.assertIn("dropped (outside every range): 1", out)

    def test_bad_rows_are_refused(self):
        good = {"id": "a", "t": "2019-01-01", "label": 1, "value": 1, "state": {}}
        for bad in ({"label": 2}, {"value": "x"}, {"t": "not a time"}, {"state": []}):
            self.tt.write_rows([{**good, **bad}])
            self.assertEqual(self.tt.plab("split").code, 2, bad)
        self.tt.write_rows([good, good])
        r = self.tt.plab("split")
        self.assertEqual(r.code, 2)
        self.assertIn("duplicate", r.err)

    def test_true_and_false_are_classes_1_and_0(self):
        rows = alert_rows(800)
        self.tt.write_rows([{**r, "label": bool(r["label"])} for r in rows])
        out = self.tt.ok("split")
        ones = sum(r["label"] for r in rows if when(r).year < 2022)
        self.assertIn(f"1: {ones} ", out.splitlines()[0])

    def test_overlapping_periods_are_refused(self):
        self.tt.write_task(alert_task(splits={"train": ["2018-01-01", "2023-01-01"], "eval": ["2022-01-01", "2024-01-01"]}))
        r = self.tt.plab("split")
        self.assertEqual(r.code, 2)
        self.assertIn("overlaps", r.err)


class TestDesign(unittest.TestCase):
    """The units a training run measures, seen through its record."""

    @classmethod
    def setUpClass(cls):
        cls.tt = TempTask()
        cls.tt.ok("split")
        cls.train = cls.tt.split_rows("train")
        cls.record = cls.tt.record("run", cls.tt.prompt("window"), "--decider", "machine")

    @classmethod
    def tearDownClass(cls):
        cls.tt.cleanup()

    def test_the_training_sample_keeps_every_positive_and_four_negatives_each(self):
        positives = sum(r["label"] for r in self.train)
        negatives = len(self.train) - positives
        self.assertEqual(self.record["class_counts"], {"0": min(negatives, 4 * positives), "1": positives})
        self.assertEqual(self.record["n"], positives + min(negatives, 4 * positives))

    def test_the_weights_give_back_the_population(self):
        population = round(1000 * sum(r["value"] for r in self.train) / len(self.train), 1)
        self.assertEqual(self.record["decision"]["all_in_value_per_1000"], population)

    def test_the_sample_is_the_same_every_time(self):
        again = self.tt.record("run", self.tt.prompt("window"), "--decider", "machine")
        for key in ("n", "class_counts", "cv_gain_pct", "auc", "decision", "xfit"):
            self.assertEqual(again[key], self.record[key], key)

    def test_the_training_split_has_two_halves(self):
        self.assertIsNotNone(self.record["auc_h1"])
        self.assertIsNotNone(self.record["auc_h2"])
        self.assertIn("xfit", self.record)


class TestGroups(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask(rows=alert_rows(800))
        self.tt.ok("split")

    def tearDown(self):
        self.tt.cleanup()

    def run_with(self, cv_groups):
        self.tt.write_task(alert_task(cv_groups=cv_groups))
        return self.tt.plab("run", self.tt.prompt("window"), "--decider", "machine", "--json")

    def test_folds_can_follow_the_calendar_a_field_or_each_row(self):
        gains = {}
        for cv_groups in ("month", "week", "day", "row", "time_utc"):
            r = self.run_with(cv_groups)
            self.assertEqual(r.code, 0, cv_groups)
            gains[cv_groups] = json.loads(r.out)["cv_gain_pct"]
        self.assertNotEqual(gains["month"], gains["row"])

    def test_a_grouping_that_puts_every_row_together_is_refused(self):
        r = self.run_with("no_such_field")
        self.assertEqual(r.code, 2)
        self.assertIn("cv_groups", r.err)


class TestFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tt = TempTask()
        cls.tt.ok("split")

    @classmethod
    def tearDownClass(cls):
        cls.tt.cleanup()

    def test_fields_lists_the_state_and_the_break_even_rate(self):
        out = self.tt.ok("fields")
        for field in ("time_utc", "x", "noise", "maybe", "series", "flag"):
            self.assertIn(field, out)
        self.assertIn("below 3.85% positive (above 96.15% negative)", out)     # 2 / (2 + 50)
        self.assertIn("lists of 12-12 values", out)

    def test_auc_of_a_single_field(self):
        out = self.tt.ok("fields", "--auc")
        x = next(line for line in out.splitlines() if line.strip().startswith("x "))
        self.assertRegex(x, r"AUC 0\.\d{3} \(halves 0\.\d{3} / 0\.\d{3}\)")
        series = next(line for line in out.splitlines() if line.strip().startswith("series "))
        self.assertNotIn("AUC", series)


if __name__ == "__main__":
    unittest.main()
