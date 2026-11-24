import math
import unittest
from unittest import mock

from promptlab.answers import Reply
from promptlab.deciders import GatewayDecider
from promptlab.questions import Refused, Yes
from tests.helpers import WINDOW, TempTask, alert_rows, alert_task

YARDSTICKS = ("cv_gain_pct", "asis_gain_pct", "auc", "mi_pct", "mi_p", "pos_by_quintile_pct", "decision", "xfit")


def in_window(row):
    return "13:00" <= row["state"]["time_utc"] <= "15:59"


class Deciders(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask(rows=alert_rows(900))
        self.tt.ok("split")
        self.window = self.tt.prompt("window")

    def tearDown(self):
        self.tt.cleanup()

    def run_fake(self, *more):
        return self.tt.record("run", self.window, "--decider", "fake", *more)

    def ledger(self, agent="lab"):
        return self.tt.log("ledger", agent)


class TestMachine(Deciders):
    def test_the_machine_is_free_and_applies_the_expr_exactly(self):
        self.tt.ok("prereg", "freeze")
        r = self.tt.record("check", self.window, "--agent", "a", "--decider", "machine")
        rows = self.tt.split_rows("eval")
        kept = [x for x in rows if not in_window(x)]
        self.assertEqual((r["calls"], r["cost_usd"]), (0, 0.0))
        self.assertEqual(r["decision"]["skip_frac"], round(1 - len(kept) / len(rows), 3))
        self.assertEqual(r["decision"]["value_per_1000"], round(1000 * sum(x["value"] for x in kept) / len(rows), 1))
        self.assertEqual(self.ledger("a"), [])

    def test_the_machine_refuses_a_rule_without_an_expr(self):
        r = self.tt.plab("run", self.tt.prompt("words"), "--decider", "machine")
        self.assertEqual(r.code, 2)
        self.assertIn("rule 1", r.err)


class TestFake(Deciders):
    def test_fake_answers_are_stored_and_the_same_every_time(self):
        first, second = self.run_fake(), self.run_fake()
        self.assertEqual(first["calls"], first["n"])
        self.assertEqual(second["calls"], 0)
        for key in YARDSTICKS:
            self.assertEqual(first[key], second[key], key)

    def test_the_fake_follows_the_machine_with_a_little_noise(self):
        machine = self.tt.record("run", self.window, "--decider", "machine")
        fake = self.run_fake()
        self.assertAlmostEqual(fake["decision"]["skip_frac"], machine["decision"]["skip_frac"], delta=0.03)
        self.assertEqual(fake["null_pct"]["kind"], "sharp")

    def test_an_interrupted_run_goes_on_where_it_stopped(self):
        self.tt.ok("agree", self.window, "--decider", "fake", "--limit", "15")
        r = self.run_fake()
        self.assertEqual(r["calls"], r["n"] - 15)

    def test_calls_are_entered_in_the_ledger_sixty_at_a_time(self):
        r = self.run_fake()
        calls = [line["calls"] for line in self.ledger()]
        self.assertEqual(sum(calls), r["n"])
        self.assertEqual(calls, [60] * (r["n"] // 60) + [r["n"] % 60])

    def test_cache_only_never_calls(self):
        r = self.tt.plab("run", self.window, "--decider", "fake", "--cache-only")
        self.assertEqual(r.code, 2)
        self.assertIn("--cache-only", r.err)
        self.assertEqual(self.ledger(), [])

    def test_the_budget_is_checked_before_any_call(self):
        task = alert_task(budget={"usd": 0.05, "eval_checks_per_agent": 2, "nominations_per_agent": 1})
        task["deciders"] = {"fake": {"kind": "fake", "cost_per_call": 0.01}}
        self.tt.write_task(task)
        r = self.tt.plab("run", self.window, "--decider", "fake")
        self.assertEqual(r.code, 2)
        self.assertIn("budget", r.err)
        self.assertEqual(self.ledger(), [])
        r = self.run_fake("--yes")
        self.assertAlmostEqual(r["cost_usd"], 0.01 * r["n"], places=6)
        self.assertAlmostEqual(sum(line["cost_usd"] for line in self.ledger()), 0.01 * r["n"], places=6)
        self.assertIn(f"spent ${0.01 * r['n']:.5f} of $0.05", self.tt.ok("cost"))


def gateway(answer_of):
    """The gateway decider, answering each ask with answer_of(decider name, ask) instead of calling out."""
    asked = []

    def answer(self, asks):
        asked.extend(asks)
        return [answer_of(self.name, a) for a in asks]
    return mock.patch.object(GatewayDecider, "answer", answer), asked


def by_window(name, ask):
    return Reply((Yes(0.9 if "13:00" <= ask.data["time_utc"] <= "15:59" else 0.1),), cost_usd=0.00002)


def minute(ask):
    return int(ask.data["time_utc"][3:])


class TestStore(Deciders):
    """What is paid for is kept, keyed by what the decider read."""

    def edited(self, **change):
        return self.tt.write_prompt("window", {**WINDOW, "veto_rules": [{**WINDOW["veto_rules"][0], **change}]})

    def test_a_model_does_not_read_the_expr_so_its_answers_outlive_an_edit_of_it(self):
        patch, asked = gateway(by_window)
        with patch:
            first = self.tt.record("run", self.window, "--decider", "jev")
            self.edited(expr="'13:00' <= time_utc <= '16:59'")
            second = self.tt.record("run", self.window, "--decider", "jev", "--cache-only")
        self.assertEqual((first["calls"], second["calls"]), (len(asked), 0))
        self.assertGreater(len(asked), 0)
        self.assertNotEqual(first["prompt_id"], second["prompt_id"])
        self.assertEqual(first["cv_gain_pct"], second["cv_gain_pct"])
        self.assertAlmostEqual(first["cost_usd"], 0.00002 * first["calls"], places=8)

    def test_a_model_reads_the_words_so_an_edit_of_them_asks_again(self):
        patch, _ = gateway(by_window)
        with patch:
            self.tt.record("run", self.window, "--decider", "jev")
            self.edited(text="(a) Skip in the early afternoon.")
            r = self.tt.plab("run", self.window, "--decider", "jev", "--cache-only")
        self.assertEqual(r.code, 2)

    def test_the_fake_reads_the_expr_so_an_edit_of_it_asks_again(self):
        self.run_fake()
        self.edited(expr="'13:00' <= time_utc <= '16:59'")
        self.assertEqual(self.tt.plab("run", self.window, "--decider", "fake", "--cache-only").code, 2)
        self.assertEqual(self.run_fake()["calls"], self.run_fake()["n"])

    def test_units_a_model_cannot_tell_apart_are_asked_once(self):
        patch, asked = gateway(by_window)
        with patch:
            r = self.tt.record("run", self.window, "--decider", "jev")
        times = {x["state"]["time_utc"] for x in self.tt.split_rows("train")}
        self.assertLess(r["calls"], r["n"])
        self.assertLessEqual(r["calls"], len(times))
        self.assertEqual(len({a.data["time_utc"] for a in asked}), len(asked))

    def test_failed_calls_are_asked_again_and_a_partial_set_is_not_measured(self):
        def half_fail(name, ask):
            return Reply(None, error="HTTP 400 bad request") if minute(ask) % 2 else by_window(name, ask)
        patch, asked = gateway(half_fail)
        with patch:
            r = self.tt.plab("run", self.window, "--decider", "jev")
        self.assertEqual(r.code, 2)
        self.assertIn("no answer", r.err)
        failed = sum(line["errors"] for line in self.ledger())
        self.assertEqual(failed, sum(minute(a) % 2 for a in asked))
        self.assertGreater(failed, 0)
        patch, _ = gateway(by_window)
        with patch:
            self.assertEqual(self.tt.record("run", self.window, "--decider", "jev")["calls"], failed)


class TestRefusal(Deciders):
    def test_a_refused_question_is_counted_and_measured_as_the_uniform_answer(self):
        """One decider refuses the inputs of every fifth minute; the other answers 0.5 there. They measure alike."""
        task = alert_task()
        task["deciders"] = {"declines": {"kind": "gateway", "model": "m/declines", "price_per_mtok": 0.042},
                            "shrugs": {"kind": "gateway", "model": "m/shrugs", "price_per_mtok": 0.042}}
        self.tt.write_task(task)
        self.tt.ok("prereg", "freeze")

        def answer_of(name, ask):
            if minute(ask) % 5:
                return by_window(name, ask)
            return Reply((Refused() if name == "declines" else Yes(0.5),), cost_usd=0.00002)
        patch, _ = gateway(answer_of)
        with patch:
            declines = self.tt.record("check", self.window, "--agent", "a", "--decider", "declines")
            shrugs = self.tt.record("check", self.window, "--agent", "a", "--decider", "shrugs")
        rows = self.tt.split_rows("eval")
        refused = sum(int(x["state"]["time_utc"][3:]) % 5 == 0 for x in rows)
        self.assertGreater(refused, 0)
        self.assertEqual((declines["refused"], shrugs["refused"]), (refused, 0))
        self.assertEqual((declines["n"], declines["missing"]), (len(rows), 0))
        for key in ("cv_gain_pct", "asis_gain_pct", "auc", "mi_pct", "mi_p", "pos_by_quintile_pct", "decision"):
            self.assertEqual(declines[key], shrugs[key], key)

    def test_a_decider_that_refuses_everything_tells_nothing(self):
        patch, _ = gateway(lambda name, ask: Reply((Refused(),)))
        with patch:
            r = self.tt.record("run", self.window, "--decider", "jev")
        self.assertEqual(r["refused"], r["n"])
        self.assertEqual(r["auc"], 0.5)
        self.assertLess(r["cv_gain_pct"], 0)
        self.assertEqual(r["asis_gain_pct"], round(100 * (1 - 1 / self.population_entropy_bits()), 3))     # every forecast is 1/2: CE = 1 bit

    def population_entropy_bits(self):
        """H(Z) of the training split, which the weights of the down-sample restore."""
        train = self.tt.split_rows("train")
        p = sum(r["label"] for r in train) / len(train)
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


if __name__ == "__main__":
    unittest.main()
