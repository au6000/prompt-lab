import json
import re
import unittest

from tests.helpers import TREND, WINDOW, TempTask, alert_rows

MEASURED = ("n", "class_counts", "cv_gain_pct", "asis_gain_pct", "auc", "auc_h1", "auc_h2", "mi_pct", "decision", "xfit", "null_pct",
            "rule_hits")


class Prompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tt = TempTask(rows=alert_rows(900))
        cls.tt.ok("split")

    @classmethod
    def tearDownClass(cls):
        cls.tt.cleanup()

    def lint(self, prompt):
        return self.tt.plab("lint", self.tt.write_prompt("p", prompt))

    def prompt_id(self, prompt):
        return re.search(r"\[([0-9a-f]{12})\]", self.tt.ok("lint", self.tt.write_prompt("p", prompt))).group(1)

    def machine(self, prompt):
        return self.tt.record("run", self.tt.write_prompt("p", prompt), "--decider", "machine")


class TestForms(Prompts):
    def test_veto_rules_are_one_rules_predicate_whose_yes_forecasts_class_1(self):
        as_questions = {"name": "window", "fields": ["time_utc"],
                        "questions": [{"name": "skip_veto", "type": "predicate", "rules": WINDOW["veto_rules"]}],
                        "forecast": {"question": "skip_veto", "classes": {"yes": "1", "no": "0"}}}
        old, new = self.machine(WINDOW), self.machine(as_questions)
        self.assertEqual(old["prompt_id"], new["prompt_id"])
        for key in MEASURED:
            self.assertEqual(old[key], new[key], key)

    def test_a_forecast_the_other_way_round_reverses_the_sign_aware_yardsticks_only(self):
        """The fake decider answers both prompts alike: they ask the same question."""
        question = {"q": {"type": "boolean", "instructions": "Will this alert fail?"}}
        runs = {}
        for high_means in ("positive", "negative"):
            path = self.tt.write_prompt(high_means, {"fields": ["x"], "questions": question, "score": {"q": "q", "high_means": high_means}})
            runs[high_means] = self.tt.record("run", path, "--decider", "fake")
        self.assertEqual(runs["positive"]["cv_gain_pct"], runs["negative"]["cv_gain_pct"])
        self.assertAlmostEqual(runs["positive"]["auc"], 1 - runs["negative"]["auc"], places=3)

    def test_a_score_forecasts_through_its_levels(self):
        """The top level of three forecasts class 1, so the machine's forecast is the trend rule's."""
        size = {"fields": ["x"], "forecast": {"question": "size", "classes": {"small": "0", "medium": "0", "large": "1"}},
                "questions": [{"name": "size", "type": "score", "instructions": "How big is x?", "levels": ["small", "medium", "large"],
                               "expr": "(x > 0) + (x > 1)"}]}
        score, veto = self.machine(size), self.machine(TREND)
        for key in ("asis_gain_pct", "auc", "auc_h1", "mi_pct", "decision", "xfit"):
            self.assertEqual(score[key], veto[key], key)
        fake = self.tt.record("run", self.tt.write_prompt("p", size), "--decider", "fake")
        self.assertEqual((fake["calls"], fake["refused"]), (fake["n"], 0))
        self.assertAlmostEqual(fake["decision"]["skip_frac"], veto["decision"]["skip_frac"], delta=0.03)

    def test_a_prompt_without_a_forecast_is_measured_by_its_information_alone(self):
        r = self.machine({"fields": ["x"], "questions": [{"name": "big", "type": "predicate", "instructions": "Is x above 1?",
                                                             "expr": "x > 1"}]})
        self.assertIn("cv_gain_pct", r)
        self.assertEqual(list(r["null_pct"]["by_kind"]["soft"]), ["cv_gain_pct"])
        for key in ("asis_gain_pct", "auc", "mi_pct", "decision", "xfit"):
            self.assertNotIn(key, r)

    def test_a_fitted_forecast_reads_every_question_and_has_no_as_is_gain(self):
        r = self.machine({"fields": ["x", "time_utc"], "forecast": "fitted", "questions": [
            {"name": "big", "type": "predicate", "instructions": "Is x above 1?", "expr": "x > 1"},
            {"name": "window", "type": "predicate", "instructions": "Is it early afternoon?", "expr": "'13:00' <= time_utc <= '15:59'"}]})
        self.assertNotIn("asis_gain_pct", r)
        self.assertEqual([len(row) for row in r["lr_coef"]], [3])
        for key in ("auc", "auc_h1", "auc_h2", "decision"):
            self.assertIsNotNone(r[key], key)
        self.assertEqual((r["mi_pct"], r["mi_p"], r["pos_by_quintile_pct"]), (None, None, None))
        self.assertNotIn("xfit", r)


class TestIdentity(Prompts):
    BASE = {"fields": ["x"], "veto_rules": [{"text": "(a) Skip when x > 1.", "expr": "x > 1"}]}

    def test_the_id_follows_what_is_asked_and_how_the_machine_reads_it(self):
        base = self.prompt_id(self.BASE)
        self.assertNotEqual(base, self.prompt_id({"fields": ["x"], "veto_rules": [{"text": "(a) Skip when x > 1.", "expr": "x >= 1"}]}))
        self.assertNotEqual(base, self.prompt_id({"fields": ["x"], "veto_rules": [{"text": "(a) Skip when x > 2.", "expr": "x > 1"}]}))
        self.assertNotEqual(base, self.prompt_id({**self.BASE, "fields": ["x", "noise"]}))

    def test_the_id_does_not_follow_the_name_or_a_note(self):
        self.assertEqual(self.prompt_id(self.BASE), self.prompt_id({**self.BASE, "name": "another name", "_note": "why I wrote it"}))


class TestRefusals(Prompts):
    def test_an_expr_may_only_read_fields_given_to_the_decider(self):
        r = self.lint({"fields": ["x"], "veto_rules": [{"text": "(a) Skip when noise > 0.5.", "expr": "noise > 0.5"}]})
        self.assertEqual(r.code, 2)
        self.assertIn("noise", r.err)

    def test_an_expr_outside_the_language_is_refused(self):
        self.assertEqual(self.lint({"fields": ["x"], "veto_rules": [{"text": "bad", "expr": "x.__class__"}]}).code, 2)

    def test_malformed_prompts_are_refused(self):
        predicate = {"name": "q", "type": "predicate", "instructions": "?"}
        for bad in ({"fields": ["x"], "veto_rules": []},
                    {"fields": ["x"], "veto_rules": ["r"] * 13},
                    {"fields": ["nope"], "veto_rules": ["r"]},
                    {"fields": ["x"], "veto_rules": ["r"], "questions": {"q": {"type": "boolean", "instructions": "?"}}},
                    {"fields": ["x"], "questions": {"q": {"type": "boolean", "instructions": "?"}}},
                    {"fields": ["x"], "questions": {"q": {"type": "boolean", "instructions": "?"}}, "score": {"q": "zz", "high_means": "positive"}},
                    {"fields": ["x"], "veto_rules": ["x" * 5000]},
                    {"fields": ["x"], "questions": {"s": {"type": "score", "instructions": "?", "criteria": ["low", "high"]}},
                     "score": {"q": "s", "high_means": "positive"}},
                    {"fields": ["x"], "questions": []},
                    {"fields": ["x"], "questions": [predicate, predicate]},
                    {"fields": ["x"], "questions": [{**predicate, "rules": ["r"]}]},
                    {"fields": ["x"], "questions": [{"name": "c", "type": "choice", "instructions": "?", "choices": ["only one"]}]},
                    {"fields": ["x"], "questions": [{"name": "s", "type": "score", "instructions": "?", "levels": ["a", "a"]}]},
                    {"fields": ["x"], "questions": [predicate], "forecast": {"question": "q", "classes": {"yes": "1"}}},
                    {"fields": ["x"], "questions": [predicate], "forecast": {"question": "q", "classes": {"yes": "1", "no": "7"}}},
                    {"fields": ["x"], "questions": [predicate], "unknown_key": 1}):
            self.assertEqual(self.lint(bad).code, 2, str(bad)[:90])

    def test_the_forms_of_version_01_and_today_load(self):
        for good in ({"fields": ["x"], "questions": {"q": {"type": "boolean", "instructions": "Will it fail?"}},
                      "score": {"q": "q", "high_means": "loss"}},
                     {"fields": ["x"], "questions": {"a": {"type": "boolean", "instructions": "?"}, "b": {"type": "boolean", "instructions": "?"}},
                      "score": {"combine": "lr"}},
                     {"fields": ["x"], "questions": {"c": {"type": "choice", "instructions": "?", "criteria": {"up": "rises", "down": "falls"}}},
                      "score": {"q": "c", "high_means": "positive", "option": "down"}},
                     {"fields": "all", "questions": [{"name": "s", "type": "score", "instructions": "How big is x?",
                                                      "levels": ["small", {"label": "large", "description": "above 1"}], "expr": "1"}]}):
            self.assertEqual(self.lint(good).code, 0, str(good)[:90])


class TestLint(Prompts):
    def test_the_three_wordings_that_lowered_agreement_are_warned_about(self):
        out = self.tt.ok("lint", self.tt.write_prompt("risky", {"fields": ["x", "noise", "series", "flag"], "veto_rules": [
            {"text": "(a) Skip when x is greater than 0.4 times noise.", "expr": "x > 0.4 * noise"},
            {"text": "(b) Skip when the maximum of series is above 2.", "expr": "max(series) > 2"},
            {"text": "(c) Skip when x > 1 and flag.", "expr": "x > 1 and flag"},
            {"text": "(d) Skip when x < -1 and flag.", "expr": "x < -1 and flag"},
            {"text": "(e) Skip when noise > 0.9 and flag.", "expr": "noise > 0.9 and flag"},
            {"text": "(f) Skip when noise < 0.1 and x > 0.", "expr": "noise < 0.1 and x > 0"}]}))
        self.assertIn("rule 1: multiplies by a number below 1", out)
        self.assertIn("rule 2: takes max() over the list field series (12 values)", out)
        self.assertIn("4 rules contain", out)

    def test_a_plain_rule_gets_no_warning(self):
        out = self.tt.ok("lint", self.tt.write_prompt("plain", {"fields": ["x"], "veto_rules": [{"text": "(a) Skip when x > 1.", "expr": "x > 1"}]}))
        self.assertIn("no warnings", out)
        self.assertNotIn("WARN", out)

    def test_what_the_machine_cannot_answer_is_noted(self):
        out = self.tt.ok("lint", self.tt.prompt("words"))
        self.assertIn("rule 1: no expr", out)

    def test_a_comparison_with_none_is_warned_about(self):
        out = self.tt.ok("lint", self.tt.write_prompt("none", {"fields": ["maybe", "x"], "questions": [
            {"name": "gone", "type": "predicate", "instructions": "Is maybe missing?", "expr": "maybe == None"},
            {"name": "skip", "type": "predicate", "rules": [{"text": "(a) Skip when maybe is given and x > 1.", "expr": "maybe != None and x > 1"},
                                                            {"text": "(b) Skip when x > 2.", "expr": "x > 2"}]}]}))
        self.assertIn("question gone: the expr compares with None", out)
        self.assertIn("rule 1: the expr compares with None", out)
        self.assertNotIn("rule 2: the expr compares with None", out)

    def test_a_none_in_a_list_literal_is_warned_about_and_a_none_inside_a_list_field_is_not(self):
        out = self.tt.ok("lint", self.tt.write_prompt("none_in", {"fields": ["maybe", "series"], "questions": [
            {"name": "listed", "type": "predicate", "instructions": "Is maybe missing or 0?", "expr": "maybe in (None, 0)"},
            {"name": "gap", "type": "predicate", "instructions": "Does series have a gap?", "expr": "None in series"}]}))
        self.assertIn("question listed: the expr compares with None", out)
        self.assertNotIn("question gap: the expr compares with None", out)

    def test_the_example_of_the_brief_is_a_prompt_the_machine_answers(self):
        brief = self.tt.ok("brief", "--agent", "a1", "--direction", "(free)")
        example = brief.split("## Prompt format\n```\n")[1].split("```")[0]
        record = self.tt.record("run", self.tt.write_prompt("example", json.loads(example.replace("FIELD_A", "x").replace("FIELD_B", "flag"))),
                                "--decider", "machine")
        self.assertEqual((record["calls"], record["missing"]), (0, 0))
        self.assertIn("auc", record)


if __name__ == "__main__":
    unittest.main()
