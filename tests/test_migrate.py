import json
import os
import shutil
import unittest
from unittest import mock

from tests.helpers import TempTask, Wire, alert_rows, alert_task, data_of, yes_reply
from tests.test_deciders import YARDSTICKS

V01_TASK = {
    "name": "old",
    "_note": "a task folder as prompt-lab 0.1 wrote it",
    "splits": {"train": ["2018-01-01", "2022-01-01"], "eval": ["2022-01-01", "2024-01-01"], "test": ["2024-01-01", "2026-01-01"]},
    "half_boundary": "2020-01-01",
    "train_sample": {"neg_per_pos": 4, "seed": 1},
    "veto_head": "Rules: ",
    "veto_tail": " Skip?",
    "selection": {"require": [{"metric": "eval.decision.gain_vs_all_in_per_1000", "op": ">", "value": 0},
                              {"metric": "eval.positives", "op": ">=", "value": 5}],
                  "rank_by": [{"metric": "eval.decision.gain_vs_all_in_per_1000"}]},
    "budget": {"usd": 1.0, "eval_checks_per_agent": 2, "nominations_per_agent": 1},
    "backend": {"name": "mechanical", "model": "typesafe-ai/jev-1.14.0"},
    "metrics": {"cv_folds": 5, "cv_reps": 1, "mi_perms": 30, "null_reps": 6, "null_seed": 1},
}


WINDOW_RULE = "(a) Skip when time_utc is between 13:00 and 15:59."


class Migrate(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask(rows=alert_rows(900), task=V01_TASK)
        self.plab_dir = self.tt.dir / ".plab"

    def tearDown(self):
        self.tt.cleanup()

    def old_file(self, name, text="{}\n"):
        self.plab_dir.mkdir(exist_ok=True)
        (self.plab_dir / name).write_text(text, encoding="utf-8")

    def test_every_command_refuses_a_01_folder_and_names_the_way_out(self):
        for command in (("split",), ("fields",), ("run", self.tt.prompt("window")), ("report",)):
            r = self.tt.plab(*command)
            self.assertEqual(r.code, 2, command)
            self.assertIn("plab migrate", r.err)

    def test_without_write_nothing_changes(self):
        before = (self.tt.dir / "task.json").read_bytes()
        self.assertIn("nothing was changed", self.tt.ok("migrate"))
        self.assertEqual((self.tt.dir / "task.json").read_bytes(), before)
        self.assertFalse((self.tt.dir / "task.v1.json").exists())

    def test_a_migrated_folder_measures_with_the_settings_01_used(self):
        self.old_file("runs.jsonl")
        self.old_file("null_train.json")
        self.tt.ok("migrate", "--write")
        self.assertEqual(json.loads((self.tt.dir / "task.v1.json").read_text(encoding="utf-8")), V01_TASK)
        task = self.tt.task()
        self.assertEqual((task["format"], task["decider"], task["_note"]), (2, "machine", V01_TASK["_note"]))
        self.assertEqual(task["train_sample"], {"reference_per_other": 4, "seed": 1})
        self.assertEqual(task["rule_frame"], {"head": "Rules: ", "tail": " Skip?"})
        self.assertEqual(task["deciders"]["jev"], {"kind": "gateway", "model": "typesafe-ai/jev-1.14.0", "price_per_mtok": 0.042})
        self.assertEqual(task["decision"], {"skip_if_prob_at_least": 0.5})
        self.assertEqual(task["selection"]["require"][1]["metric"], "eval.class_counts.1")
        self.assertEqual(task["selection"]["rank_by"], [{"metric": "eval.decision.gain_vs_all_in_per_1000", "order": "desc"}])
        self.assertEqual(sorted(p.name for p in (self.plab_dir / "archive-v0.1").iterdir()), ["null_train.json", "runs.jsonl"])
        self.tt.ok("split")
        record = self.tt.record("run", self.tt.prompt("window"))
        self.assertEqual((record["decider"], record["calls"]), ("machine", 0))
        self.assertIn("nothing to do", self.tt.ok("migrate", "--write"))

    def test_a_study_frozen_under_01_is_refused_unless_it_is_unfrozen(self):
        for name in ("prereg.lock", "prereg.frozen.md", "checks.jsonl", "nominations.jsonl", "selected.json"):
            self.old_file(name)
        r = self.tt.plab("migrate", "--write")
        self.assertEqual(r.code, 2)
        self.assertIn("--unfreeze", r.err)
        self.assertNotIn("format", self.tt.task())
        self.tt.ok("migrate", "--write", "--unfreeze")
        self.assertEqual(sorted(p.name for p in (self.plab_dir / "archive-v0.1").iterdir()),
                         ["checks.jsonl", "nominations.jsonl", "prereg.frozen.md", "prereg.lock", "selected.json"])
        self.tt.ok("split")
        r = self.tt.plab("check", self.tt.prompt("window"), "--agent", "a")
        self.assertEqual(r.code, 2)
        self.assertIn("not frozen", r.err)

    def test_a_study_whose_test_was_started_is_never_unfrozen(self):
        self.old_file("prereg.lock")
        self.old_file("test.lock")
        r = self.tt.plab("migrate", "--write", "--unfreeze")
        self.assertEqual(r.code, 2)
        self.assertNotIn("format", self.tt.task())
        self.assertTrue((self.plab_dir / "prereg.lock").exists())


def windowed(state):
    return 0.9 if "13:00" <= state["time_utc"] <= "15:59" else 0.1


QUEUE = {"queue": {"type": "choice", "instructions": "Keep or skip this alert?", "criteria": {"keep": "take it", "skip": "skip it"}}}


def choose(body):
    return 200, {"model": body["model"], "provider_metadata": {"gateway": {"cost": "0.00001"}},
                 "answers": [{"type": "choice", "name": "queue", "choice": "skip",
                              "probabilities": [{"value": "keep", "probability": 0.3}, {"value": "skip", "probability": 0.7}]}]}


class MigrateAnswers(unittest.TestCase):
    """The answers 0.1 paid for move into the store, and measure as the same answers fetched today."""

    def setUp(self):
        self.tt = TempTask(rows=alert_rows(900), task=alert_task())
        self.addCleanup(self.tt.cleanup)
        self.plab_dir = self.tt.dir / ".plab"
        self.tt.ok("split")
        self.tt.write_task(V01_TASK)
        self.train = self.tt.split_rows("train")
        env = mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "sk-test"})
        env.start()
        self.addCleanup(env.stop)

    def cache_01(self, key, questions, fields, answer_of, rows):
        """A shelf of the 0.1 answer cache, as 0.1 wrote it."""
        shelf = self.plab_dir / "cache" / "jev"
        shelf.mkdir(parents=True, exist_ok=True)
        (shelf / f"{key}.meta.json").write_text(json.dumps({
            "name": key, "prompt_id": key, "model": V01_TASK["backend"]["model"], "fields": fields, "questions": questions,
            "state_extra": None, "created": "2026-11-19T00:00:00"}), encoding="utf-8")
        (shelf / f"{key}.jsonl").write_text("".join(json.dumps({"id": r["id"], "a": answer_of(r)}) + "\n" for r in rows), encoding="utf-8")

    def test_answers_and_spending_of_01_carry_over(self):
        rows = [*self.train, *self.tt.split_rows("eval")]
        skip = {"skip_veto": {"type": "boolean", "instructions": "Rules: " + WINDOW_RULE + " Skip?"}}
        self.cache_01("w1", skip, ["time_utc"], lambda r: {"skip_veto": windowed(r["state"])}, rows)
        (self.plab_dir / "ledger.jsonl").write_text(json.dumps({"time": "2026-11-19T00:00:00Z", "cmd": "run", "agent": "a", "split": "train",
                                                                "backend": "jev", "calls": 3, "errors": 0, "cost_usd": 0.25}) + "\n",
                                                    encoding="utf-8")
        self.assertIn(f"{len(rows)} answers move into the store", self.tt.ok("migrate", "--write"))
        self.assertEqual(sorted(p.name for p in (self.plab_dir / "archive-v0.1").iterdir()), ["cache", "ledger.jsonl"])
        migrated = self.tt.record("run", self.tt.prompt("window"), "--decider", "jev", "--cache-only")
        self.assertIn("spent $0.25000", self.tt.ok("cost"))
        shutil.rmtree(self.plab_dir / "answers")
        wire = Wire(lambda body: (200, yes_reply(body, windowed(data_of(body)), cost="0")))
        with wire.patch():
            fetched = self.tt.record("run", self.tt.prompt("window"), "--decider", "jev")
        self.assertEqual((migrated["calls"], fetched["calls"]), (0, len(wire.requests)))
        self.assertGreater(fetched["calls"], 0)
        for key in YARDSTICKS:
            self.assertEqual(migrated[key], fetched[key], key)

    def test_a_choice_moves_when_01_kept_every_probability_and_a_score_never_does(self):
        short = next(r for r in self.train if r["label"] == 1)
        self.cache_01("c1", QUEUE, ["time_utc", "x"],
                      lambda r: {"queue": {"keep": 1.0} if r is short else {"keep": 0.3, "skip": 0.7}}, self.train)
        self.cache_01("s1", {"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}}, ["time_utc"],
                      lambda r: {"urgency": 0.5}, self.train)
        out = self.tt.ok("migrate", "--write")
        self.assertIn(f"cache/jev/c1: {len(self.train) - 1} answers move into the store, marked via evaluate; 1 stay behind", out)
        self.assertIn("cache/jev/s1: not moved: question urgency is a score", out)
        prompt = self.tt.write_prompt("queue", {"name": "queue", "fields": ["time_utc", "x"], "questions": QUEUE,
                                                "score": {"q": "queue", "high_means": "positive", "option": "skip"}})
        r = self.tt.plab("run", prompt, "--decider", "jev", "--cache-only")
        self.assertEqual(r.code, 2)
        self.assertIn("1 of", r.err)
        wire = Wire(choose)
        with wire.patch():
            record = self.tt.record("run", prompt, "--decider", "jev")
        self.assertEqual((record["calls"], [data_of(q.body)["x"] for q in wire.requests]), (1, [short["state"]["x"]]))
        self.assertEqual(record["answers_via"], {"/v1/evaluate": record["n"] - 1, "/v1/decisions": 1})
        self.assertIn("answers from two endpoints", self.tt.ok("run", prompt, "--decider", "jev", "--cache-only"))


if __name__ == "__main__":
    unittest.main()
