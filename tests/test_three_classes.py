"""A task with three outcome classes and a prompt with two questions, from the split to the test."""
import json
import unittest

from tests.helpers import ROUTE, TempTask, ticket_rows, ticket_task


class ThreeClasses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tt = TempTask(rows=ticket_rows(), task=ticket_task(), prompts={"route": ROUTE})
        cls.route = cls.tt.prompt("route")
        cls.tt.ok("split")
        cls.train = cls.tt.record("run", cls.route, "--agent", "ana")

    @classmethod
    def tearDownClass(cls):
        cls.tt.cleanup()

    def test_1_the_training_run_measures_information_and_no_two_class_yardstick(self):
        r = self.train
        rows = self.tt.split_rows("train")
        self.assertEqual(r["n"], len(rows))
        self.assertEqual(r["class_counts"], {c: sum(x["label"] == c for x in rows) for c in ("other", "billing", "shipping")})
        self.assertGreater(r["cv_gain_pct"], 20)
        self.assertGreater(r["asis_gain_pct"], 0)
        self.assertEqual(r["null_pct"]["cv_gain_pct"], 100.0)
        self.assertEqual(list(r["null_pct"]["by_kind"]["soft"]), ["cv_gain_pct"])
        for key in ("auc", "mi_pct", "decision", "xfit", "rule_hits"):
            self.assertNotIn(key, r)
        self.assertTrue((self.tt.dir / ".plab" / "nulls" / "train.c3_p2.json").exists())

    def test_2_the_machine_reads_the_same_prompt_for_free(self):
        r = self.tt.record("run", self.route, "--decider", "machine", "--agent", "ana")
        self.assertEqual(r["calls"], 0)
        self.assertGreater(r["cv_gain_pct"], 20)
        self.assertLess(r["asis_gain_pct"], 0)       # a named option without probabilities is a point mass: wrong once, it costs dearly
        self.assertEqual(r["null_pct"]["kind"], "sharp")

    def test_2b_agree_and_drift_read_any_question_the_machine_answers(self):
        self.tt.ok("agree", self.route, "--decider", "fake", "--agent", "ana")
        found = self.tt.log("agree", "ana")[-1]["questions"]
        self.assertEqual(list(found), ["queue", "big"])
        self.assertIn("disagreements", found["queue"])
        self.assertGreater(found["queue"]["agree_pct"], 90)
        self.assertIn("both_skip", found["big"])
        self.assertEqual((found["queue"]["model_refused"], found["queue"]["machine_refused"]), (0, 0))
        out = self.tt.ok("drift", self.route, "--decider", "machine", "--by", "year")
        self.assertIn("question queue", out)
        self.assertIn("classes %", out)
        self.assertIn("billing", out)

    def test_3_freeze_check_nominate_select_test(self):
        self.tt.ok("prereg", "freeze")
        check = self.tt.record("check", self.route, "--agent", "ana")
        self.assertEqual((check["check_no"], check["split"]), (1, "eval"))
        self.assertGreater(check["cv_gain_pct"], 0)
        self.tt.ok("nominate", self.route, "--agent", "ana")
        self.assertIn("selected: route", self.tt.ok("select"))
        out = self.tt.ok("test", "--final")
        self.assertIn("cv_gain", out)
        result = json.loads((self.tt.dir / ".plab" / "test_result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["info"]["n"], len(self.tt.split_rows("test")))
        self.assertGreater(result["info"]["cv_gain_pct"], 0)
        self.assertGreater(result["info"]["asis_gain_pct"], 0)
        self.assertEqual(result["info"]["null_pct"]["cv_gain_pct"], 100.0)
        for table in ("selected", "take_everything", "threshold"):
            self.assertNotIn(table, result)
        self.assertEqual(self.tt.plab("test", "--final").code, 2)
        report = self.tt.record("report")
        self.assertEqual(report["eval_looks_by_agent"], {"ana": 1, "select": 1})
        self.assertEqual(report["phase"], "closed")


class ThreeClassTasks(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask(rows=ticket_rows(600), task=ticket_task(), prompts={"route": ROUTE})

    def tearDown(self):
        self.tt.cleanup()

    def refused(self, task, saying):
        self.tt.write_task(task)
        r = self.tt.plab("split")
        self.assertEqual(r.code, 2)
        self.assertIn(saying, r.err)

    def test_a_selection_rule_must_read_what_the_task_can_produce(self):
        task = ticket_task()
        del task["selection"]
        self.refused(task, "eval.decision.value_per_1000")
        self.refused(ticket_task(selection={"require": [{"metric": "eval.auc", "op": ">", "value": 0.5}], "rank_by": []}), "two classes")
        self.refused(ticket_task(selection={"require": [{"metric": "eval.cv_gain", "op": ">", "value": 0}], "rank_by": []}), "eval.cv_gain")
        self.tt.write_task(ticket_task(selection={"require": [{"metric": "eval.class_counts.billing", "op": ">=", "value": 30}],
                                                  "rank_by": [{"metric": "eval.asis_gain_pct", "order": "desc"}]}))
        self.tt.ok("split")

    def test_the_veto_use_needs_two_classes_and_a_value(self):
        self.refused(ticket_task(decision={"skip_if_prob_at_least": 0.5}), "decision")

    def test_an_outcome_outside_the_classes_is_refused(self):
        self.tt.write_rows([*ticket_rows(50), {"id": "x", "t": "2019-01-01", "state": {}, "queue": "returns"}])
        r = self.tt.plab("split")
        self.assertEqual(r.code, 2)
        self.assertIn("returns", r.err)

    def test_veto_rules_are_refused_where_there_is_no_one_positive_class(self):
        self.tt.ok("split")
        path = self.tt.write_prompt("veto", {"fields": ["x"], "veto_rules": [{"text": "(a) Skip when x > 1.", "expr": "x > 1"}]})
        r = self.tt.plab("lint", path)
        self.assertEqual(r.code, 2)
        self.assertIn("two-class", r.err)


if __name__ == "__main__":
    unittest.main()
