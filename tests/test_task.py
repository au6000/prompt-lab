import unittest

from promptlab.task import NOT_TRAIN, SELECTABLE, TRAIN_ONLY
from tests.helpers import TempTask, alert_rows, alert_task

MISSING = object()


def at(record, path):
    """What a record holds at a dotted path; MISSING when it has no such path."""
    for part in path.split("."):
        if not isinstance(record, dict) or part not in record:
            return MISSING
        record = record[part]
    return record


class TestTaskJson(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask(rows=alert_rows(300))

    def tearDown(self):
        self.tt.cleanup()

    def refused(self, saying, **change):
        self.tt.write_task(alert_task(**change))
        r = self.tt.plab("split")
        self.assertEqual(r.code, 2, change)
        self.assertIn(saying, r.err)

    def test_misspelt_and_misshapen_settings_are_refused_with_their_name(self):
        self.refused("cv_group", cv_group="week")
        self.refused("budget.usd", budget={"usd": "a lot"})
        self.refused("metrics.cv_folds", metrics={"cv_folds": 1, "cv_reps": 1, "mi_perms": 30, "null_reps": 6, "null_seed": 1})
        self.refused("train_sample.reference_per_other", train_sample={"reference_per_other": 0, "seed": 1})
        self.refused("splits", splits={"train": ["2018-01-01", "2022-01-01"]})
        self.refused("decider", decider="nobody")
        self.refused("deciders.mine", deciders={"mine": {"kind": "gateway", "model": "a/b"}})
        self.refused("outcome", outcome={"classes": ["only"]})

    def test_the_veto_use_needs_a_value(self):
        self.refused("decision", rows={"value": None})

    def test_a_note_is_not_a_setting(self):
        self.tt.write_task(alert_task(_note="why this task exists"))
        self.tt.ok("split")

    def test_a_selection_rule_names_numbers_a_measurement_holds(self):
        self.refused("eval.xfit.value_per_1000", selection={"require": [{"metric": "eval.xfit.value_per_1000", "op": ">", "value": 0}],
                                                            "rank_by": []})
        self.refused("selection.require[0].op", selection={"require": [{"metric": "eval.auc", "op": "=>", "value": 0}], "rank_by": []})
        self.refused("half_boundary", half_boundary=None,
                     selection={"require": [{"metric": "train.xfit.value_per_1000", "op": ">", "value": 0}], "rank_by": []})
        self.refused(NOT_TRAIN, selection={"require": [{"metric": "train.transfer_gain_pct", "op": ">", "value": 0}], "rank_by": []})
        self.tt.write_task(alert_task(selection={"require": [{"metric": "eval.transfer_gain_pct", "op": ">", "value": 0}],
                                                 "rank_by": [{"metric": "eval.transfer_gain_pct", "order": "desc"}]}))
        self.tt.ok("split")


class TestSelectableNumbers(unittest.TestCase):
    """Every number a selection rule may name is where a measurement puts it."""

    @classmethod
    def setUpClass(cls):
        cls.tt = TempTask()
        cls.tt.ok("split")
        window = cls.tt.prompt("window")
        cls.train = cls.tt.record("run", window, "--decider", "machine")
        cls.tt.ok("prereg", "freeze")
        cls.eval = cls.tt.record("check", window, "--decider", "machine", "--agent", "a")

    @classmethod
    def tearDownClass(cls):
        cls.tt.cleanup()

    def test_on_the_training_split(self):
        for name, needs in SELECTABLE.items():
            if NOT_TRAIN in needs:
                self.assertIs(at(self.train, name), MISSING, name)
            else:
                self.assertIsNot(at(self.train, name), MISSING, name)
        self.assertIsInstance(at(self.train, "decision.value_per_1000"), float)

    def test_on_the_evaluation_split(self):
        for name, needs in SELECTABLE.items():
            if TRAIN_ONLY in needs:
                self.assertIn(at(self.eval, name), (MISSING, None), name)
            else:
                self.assertIsNot(at(self.eval, name), MISSING, name)

if __name__ == "__main__":
    unittest.main()
