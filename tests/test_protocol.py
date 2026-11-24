import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from promptlab import __version__, protocol
from promptlab.layout import PlabError
from promptlab.study import Study as PlabStudy
from tests.helpers import TempTask, alert_task, plab

MACHINE = ("--decider", "machine")


class Study(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask()
        self.tt.ok("split")
        self.window, self.trend = self.tt.prompt("window"), self.tt.prompt("trend")

    def tearDown(self):
        self.tt.cleanup()

    def refused(self, *args, saying=None):
        r = self.tt.plab(*args)
        self.assertEqual(r.code, 2, args)
        if saying:
            self.assertIn(saying, r.err)

    def check(self, prompt, agent="a1"):
        return self.tt.record("check", prompt, "--agent", agent, *MACHINE)

    def plab_dir(self):
        return self.tt.dir / ".plab"


class TestBeforeTheFreeze(Study):
    def test_a_check_needs_a_frozen_preregistration(self):
        self.refused("check", self.window, "--agent", "a1", *MACHINE, saying="not frozen")

    def test_evaluation_rows_are_not_read_without_a_lock(self):
        self.refused("null", self.window, "--split", "eval", saying="not frozen")
        self.assertEqual(list((self.plab_dir() / "nulls").glob("eval.*")) if (self.plab_dir() / "nulls").exists() else [], [])
        self.tt.ok("null", self.window, "--split", "train")
        self.tt.ok("prereg", "freeze")
        self.assertIn("zero point on eval", self.tt.ok("null", self.window, "--split", "eval"))

    def test_the_study_hands_out_evaluation_and_test_rows_only_for_a_lock(self):
        study = PlabStudy(self.tt.dir)
        for split in ("eval", "test"):
            with self.assertRaises(PlabError):
                study.design(split)
        self.tt.ok("prereg", "freeze")
        lock = protocol.enter(study, "check")
        self.assertEqual(len(study.design("eval", lock).rows), len(self.tt.split_rows("eval")))

    def test_the_template_preregistration_is_refused(self):
        (self.tt.dir / "prereg.md").write_text("<!-- plab:template -->\n# todo\n", encoding="utf-8")
        self.refused("prereg", "freeze", saying="template")

    def test_nothing_is_selected_or_tested_before_the_freeze(self):
        for command in (("nominate", self.window, "--agent", "a1"), ("select",), ("test", "--final")):
            self.refused(*command, saying="not frozen")


class TestFullStudy(Study):
    def test_freeze_check_nominate_select_test_report(self):
        self.assertIn("not frozen", self.tt.ok("prereg", "status"))
        self.tt.ok("prereg", "freeze")
        self.assertIn("unchanged", self.tt.ok("prereg", "status"))
        self.refused("split", "--force", saying="frozen")
        for name in ("select", "white space", "..", None):
            self.refused("check", self.window, *(("--agent", name) if name else ()), *MACHINE, saying="--agent")

        self.assertEqual((self.check(self.trend)["check_no"], self.check(self.window)["check_no"]), (1, 2))
        self.refused("check", self.window, "--agent", "a1", *MACHINE, saying="2/2")
        self.refused("prereg", "freeze", "--force", saying="looked at")

        self.refused("test", "--final", saying="nothing selected")
        self.tt.ok("nominate", self.window, "--agent", "a1", *MACHINE)
        self.refused("nominate", self.window, "--agent", "a1", *MACHINE, saying="already nominated")
        self.refused("nominate", self.trend, "--agent", "a1", *MACHINE, saying="1/1")
        self.tt.ok("nominate", self.trend, "--agent", "a2", *MACHINE)
        self.assertIn("selected: window", self.tt.ok("select"))
        selected = json.loads((self.plab_dir() / "selected.json").read_text(encoding="utf-8"))
        self.assertEqual([c["name"] for c in selected["candidates"]], ["window", "trend"])
        self.assertEqual(selected["selected"]["nominated_by"], "a1")

        self.refused("test", saying="--final")
        self.tt.ok("test", "--final")
        result = json.loads((self.plab_dir() / "test_result.json").read_text(encoding="utf-8"))
        test_rows = self.tt.split_rows("test")
        self.assertEqual(result["take_everything"]["taken"], len(test_rows))
        self.assertEqual(result["take_everything"]["value_total"], round(sum(r["value"] for r in test_rows), 1))
        kept = [r for r in test_rows if not "13:00" <= r["state"]["time_utc"] <= "15:59"]
        self.assertEqual(result["selected"]["taken"], len(kept))
        self.assertEqual(result["selected"]["value_total"], round(sum(r["value"] for r in kept), 1))
        self.assertIn("cv_gain_pct", result["info"]["null_pct"])

        self.refused("test", "--final", saying="closed")
        self.refused("check", self.window, "--agent", "a2", *MACHINE, saying="closed")
        self.refused("select", saying="closed")
        report = self.tt.record("report")
        self.assertEqual(report["eval_looks_by_agent"], {"a1": 2, "select": 2})
        self.assertEqual(report["eval_looks_total"], 4)
        self.assertEqual(report["selected"]["name"], "window")
        self.assertEqual(report["test"], result)
        self.assertIn("looked at 4 times", self.tt.ok("report"))

    def test_a_check_measures_in_the_training_direction_and_a_run_does_not(self):
        train = self.tt.record("run", self.window, *MACHINE)
        self.assertNotIn("transfer_gain_pct", train)
        self.assertNotIn("transfer_gain_pct", train["null_pct"])
        self.tt.ok("prereg", "freeze")
        held_out = self.check(self.window)
        self.assertIsInstance(held_out["transfer_gain_pct"], float)
        self.assertIsInstance(held_out["null_pct"]["transfer_gain_pct"], float)
        self.assertIn("transfer%", self.tt.ok("board", "--split", "eval", "--agent", "a1"))

    def test_evaluation_has_no_halves(self):
        self.tt.ok("prereg", "freeze")
        r = self.check(self.window)
        self.assertEqual((r["split"], r["auc_h1"], r["auc_h2"]), ("eval", None, None))
        self.assertNotIn("xfit", r)
        self.assertNotIn("rule_hits", r)
        self.assertEqual(r["n"], len(self.tt.split_rows("eval")))

    def test_a_fitted_forecast_is_fitted_on_training_answers_alone(self):
        fitted = self.tt.write_prompt("fitted", {"fields": ["x", "time_utc"], "forecast": "fitted", "questions": [
            {"name": "big", "type": "predicate", "instructions": "Is x above 1?", "expr": "x > 1"},
            {"name": "window", "type": "predicate", "instructions": "Is it early afternoon?", "expr": "'13:00' <= time_utc <= '15:59'"}]})
        train = self.tt.record("run", fitted, *MACHINE)
        self.tt.ok("prereg", "freeze")
        held_out = self.check(fitted)
        self.assertEqual(held_out["lr_coef"], train["lr_coef"])
        self.assertIn("decision", held_out)
        self.assertNotIn("asis_gain_pct", held_out)

    def test_a_check_that_fails_costs_no_look(self):
        self.tt.ok("prereg", "freeze")
        self.refused("check", self.window, "--agent", "a1", "--decider", "fake", "--cache-only", saying="--cache-only")
        self.assertEqual(self.check(self.window)["check_no"], 1)

    def test_two_checks_of_one_agent_do_not_run_at_once(self):
        self.tt.ok("prereg", "freeze")
        running = self.plab_dir() / "agents" / "a1" / "check.lock"
        running.parent.mkdir(parents=True)
        running.write_text("", encoding="utf-8")
        self.refused("check", self.window, "--agent", "a1", *MACHINE, saying="another check")
        self.assertEqual(self.check(self.window, agent="a2")["check_no"], 1)
        running.unlink()
        self.assertEqual(self.check(self.window)["check_no"], 1)
        self.assertFalse(running.exists())

    def test_no_selection_keeps_the_test_unused(self):
        self.tt.write_task(alert_task(selection={"require": [{"metric": "eval.cv_gain_pct", "op": ">", "value": 1000}], "rank_by": []}))
        self.tt.ok("prereg", "freeze")
        self.tt.ok("nominate", self.window, "--agent", "a1", *MACHINE)
        self.assertIn("selected: none", self.tt.ok("select"))
        self.refused("test", "--final", saying="no prompt")
        self.assertFalse((self.plab_dir() / "test.lock").exists())

    def test_a_test_that_did_not_finish_is_finished_with_resume(self):
        task = alert_task(budget={"usd": 100.0, "eval_checks_per_agent": 2, "nominations_per_agent": 1},
                          selection={"require": [{"metric": "eval.n", "op": ">", "value": 0}], "rank_by": []})
        task["deciders"] = {"fake": {"kind": "fake", "cost_per_call": 0.01}}
        self.tt.write_task(task)
        self.tt.ok("prereg", "freeze")
        self.tt.ok("nominate", self.window, "--agent", "a1", "--decider", "fake")
        self.tt.ok("select")
        self.tt.write_task({**task, "budget": {**task["budget"], "usd": 0.0}})
        self.refused("test", "--final", saying="budget")
        self.refused("test", "--final", saying="--resume")
        self.refused("check", self.window, "--agent", "a1", *MACHINE, saying="--resume")
        self.tt.write_task(task)
        self.tt.ok("test", "--final", "--resume")
        self.refused("test", "--final", "--resume", saying="closed")


class TestTampering(Study):
    def setUp(self):
        super().setUp()
        self.tt.ok("prereg", "freeze")

    def edit_task(self, change):
        task = self.tt.task()
        change(task)
        self.tt.write_task(task)

    def test_an_edited_preregistration_is_refused_until_it_is_put_back(self):
        path = self.tt.dir / "prereg.md"
        frozen = path.read_text(encoding="utf-8")
        path.write_text(frozen + "\nchanged my mind\n", encoding="utf-8")
        self.refused("check", self.window, "--agent", "a1", *MACHINE, saying="prereg.md")
        self.assertIn("CHANGED", self.tt.ok("prereg", "status"))
        path.write_text(frozen, encoding="utf-8")
        self.assertEqual(self.check(self.window)["check_no"], 1)

    def test_an_edited_selection_rule_is_refused(self):
        self.edit_task(lambda t: t["selection"]["require"][0].update(value=-1e9))
        self.refused("select", saying="selection")

    def test_another_model_under_a_decider_name_is_refused(self):
        self.edit_task(lambda t: t.update(deciders={"jev": {"model": "someone/else"}}))
        self.refused("check", self.window, "--agent", "a1", *MACHINE, saying="deciders")

    def test_the_budget_in_money_and_the_prices_may_change(self):
        self.edit_task(lambda t: t["budget"].update(usd=7.0))
        self.edit_task(lambda t: t.update(deciders={"jev": {"price_per_mtok": 0.05}}))
        self.assertEqual(self.check(self.window)["check_no"], 1)

    def test_edited_evaluation_rows_are_refused(self):
        path = self.plab_dir() / "data" / "eval.jsonl"
        frozen = path.read_bytes()
        path.write_bytes(frozen.replace(b'"label": 1', b'"label": 0', 1))
        self.refused("check", self.window, "--agent", "a1", *MACHINE, saying="eval.jsonl")
        path.write_bytes(frozen)
        self.assertEqual(self.check(self.window)["check_no"], 1)
        self.assertEqual(len(self.tt.log("checks", "a1")), 1)


class TestOnTheTrainingSplit(Study):
    def test_explore_commands(self):
        record = self.tt.record("run", self.window, *MACHINE, "--agent", "a1")
        self.assertEqual(record["split"], "train")
        for key in ("xfit", "null_pct", "rule_hits", "decision"):
            self.assertIn(key, record)
        self.tt.ok("agree", self.window, "--decider", "fake", "--per-rule", "--agent", "a1")
        agreement = self.tt.log("agree", "a1")[-1]
        self.assertEqual(list(agreement["questions"]), ["rule_1"])
        self.assertGreater(agreement["questions"]["rule_1"]["agree_pct"], 95)
        drift = [line.split() for line in self.tt.ok("drift", self.trend, "--by", "year", *MACHINE).splitlines()[2:]]
        self.assertEqual([cells[0] for cells in drift], ["2018", "2019", "2020", "2021"])
        self.assertEqual(sum(int(cells[1]) for cells in drift), len(self.tt.split_rows("train")))     # the machine is free: every row
        self.assertIn("window", self.tt.ok("board", "--split", "train"))
        self.refused("board", "--split", "eval", saying="--agent")
        self.assertIn("at most 1 candidates", self.tt.ok("brief", "--agent", "a1", "--direction", "time windows"))
        self.assertIn("spent $0.00000", self.tt.ok("cost"))

    def test_run_with_several_deciders_compares_them(self):
        out = self.tt.ok("run", self.window, "--decider", "machine", "--decider", "fake", "--agent", "a1")
        self.assertRegex(out, r"(?m)^machine ")
        self.assertRegex(out, r"(?m)^fake ")
        self.assertEqual([r["decider"] for r in self.tt.log("runs", "a1")], ["machine", "fake"])


class TestSurveyOfEverySplit(Study):
    """`plab survey --splits all` reads the evaluation and test rows, so it waits for the closed study."""

    ONLY_CLOSED = "only once the test has run"

    def survey_refused_and_training_survey_runs(self):
        self.refused("survey", "--splits", "all", saying=self.ONLY_CLOSED)
        self.refused("survey", self.window, "--splits", "all", "--json", saying=self.ONLY_CLOSED)
        self.assertEqual(self.tt.record("survey")["splits"], ["train"])
        self.assertEqual(self.tt.log("surveys", "lab"), [])

    def test_every_split_is_refused_in_every_phase_before_the_close(self):
        task = alert_task(budget={"usd": 100.0, "eval_checks_per_agent": 2, "nominations_per_agent": 1},
                          selection={"require": [{"metric": "eval.n", "op": ">", "value": 0}], "rank_by": []})
        task["deciders"] = {"fake": {"kind": "fake", "cost_per_call": 0.01}}
        self.tt.write_task(task)
        phases = []
        for step in (lambda: None, lambda: self.tt.ok("prereg", "freeze"),
                     lambda: (self.tt.ok("nominate", self.window, "--agent", "a1", "--decider", "fake"), self.tt.ok("select")),
                     lambda: (self.tt.write_task({**task, "budget": {**task["budget"], "usd": 0.0}}),
                              self.refused("test", "--final", saying="budget"))):
            step()
            phases.append(protocol.phase(PlabStudy(self.tt.dir).layout))
            self.survey_refused_and_training_survey_runs()
        self.assertEqual(phases, ["open", "frozen", "selected", "testing"])

    def close(self):
        self.tt.ok("prereg", "freeze")
        self.tt.ok("nominate", self.window, "--agent", "a1", *MACHINE)
        self.tt.ok("select")
        self.tt.ok("test", "--final")

    def test_after_the_test_every_split_is_surveyed_as_exploratory_and_logged(self):
        self.close()
        lines = self.tt.ok("survey", self.trend, "--splits", "all", "--agent", "a1").splitlines()
        self.assertTrue(lines[0].startswith("EXPLORATORY:"), lines[0])
        self.assertTrue(lines[-1].startswith("EXPLORATORY") and "surveys.jsonl" in lines[-1], lines[-1])
        found = self.tt.record("survey", self.trend, "--splits", "all", "--agent", "a1")
        rows = {s: self.tt.split_rows(s) for s in ("train", "eval", "test")}
        self.assertEqual((found["exploratory"], found["splits"], found["units"]), (True, ["train", "eval", "test"], sum(map(len, rows.values()))))
        logged = self.tt.log("surveys", "a1")
        self.assertEqual(len(logged), 2)
        line = logged[-1]
        self.assertEqual(line["data_sha256"], {s: hashlib.sha256((self.plab_dir() / "data" / f"{s}.jsonl").read_bytes()).hexdigest()
                                               for s in rows})
        self.assertEqual(line["factors_from"]["prompt_id"], self.tt.record("run", self.trend, *MACHINE)["prompt_id"])
        self.assertEqual((line["by"], line["gap_months"], line["windows_months"], line["harness_version"]),
                         ("half", 0, [6, 12, 24, None], __version__))
        self.assertEqual(self.tt.record("report")["exploratory_surveys"], 2)
        self.assertIn("exploratory surveys of every split (after the test): 2", self.tt.ok("report"))
        self.assertEqual(self.tt.record("survey")["exploratory"], False)
        self.assertEqual(len(self.tt.log("surveys", "lab")), 0)

    def test_after_the_close_an_edited_evaluation_row_is_refused(self):
        self.close()
        path = self.plab_dir() / "data" / "eval.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"label": 1', b'"label": 0', 1))
        self.refused("survey", "--splits", "all", saying="eval.jsonl")
        self.assertEqual(self.tt.log("surveys", "lab"), [])


class TestCommandLine(unittest.TestCase):
    def test_init_makes_a_task_folder_that_loads_and_a_template_that_is_not_frozen(self):
        folder = Path(tempfile.mkdtemp(prefix="plab_init_")) / "t"
        try:
            self.assertEqual(plab("init", str(folder)).code, 0)
            self.assertEqual(json.loads((folder / "task.json").read_text(encoding="utf-8"))["format"], 2)
            self.assertIn("<!-- plab:template -->", (folder / "prereg.md").read_text(encoding="utf-8"))
            self.assertEqual(plab("init", str(folder)).code, 2)
            self.assertEqual(plab("-C", str(folder), "cost").code, 0)
        finally:
            shutil.rmtree(folder.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
