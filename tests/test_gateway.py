import itertools
import json
import os
import tempfile
import unittest
from email.utils import formatdate
from pathlib import Path
from unittest import mock

from promptlab import gateway
from promptlab.questions import Chose, Refused, Scored, Yes
from promptlab.study import Study
from promptlab.task import GatewaySettings
from tests.helpers import TempTask, Wire, alert_task, data_of, ticket_rows, ticket_task, yes_reply

THREE = {"name": "three", "fields": ["topic", "x"],
         "questions": [{"name": "queue", "type": "choice", "instructions": "Which queue?", "choices": ["other", "billing", "shipping"]},
                       {"name": "urgency", "type": "score", "instructions": "How urgent?", "levels": ["low", "medium", "high"]},
                       {"name": "big", "type": "predicate", "instructions": "Is x above 1?"}]}


def recorded(body):
    """A reply in the shape the Decisions API documents, with the options listed out of order."""
    return {"model": body["model"],
            "answers": [{"type": "choice", "name": "queue", "choice": "billing", "confidence": 0.7,
                         "probabilities": [{"value": "shipping", "probability": 0.1}, {"value": "other", "probability": 0.2},
                                           {"value": "billing", "probability": 0.7}]},
                        {"type": "score", "name": "urgency", "score": 1.3, "confidence": 0.54,
                         "probabilities": [{"value": 2, "label": "high", "probability": 0.38}, {"value": 0, "label": "low", "probability": 0.08},
                                           {"value": 1, "label": "medium", "probability": 0.54}]},
                        {"type": "predicate", "name": "big", "probability": 0.95}],
            "usage": {"input_tokens": 96, "output_tokens": 0, "total_tokens": 96},
            "provider_metadata": {"gateway": {"routing": {"originalModelId": "typesafe-ai/jev", "finalProvider": "typesafe-ai"},
                                              "cost": "0.0000096", "generationId": "gen_01"}}}


def changed(k, **item):
    """recorded(), with answer k replaced."""
    def reply(body):
        r = recorded(body)
        r["answers"][k] = {"name": r["answers"][k]["name"], "type": r["answers"][k]["type"], **item}
        return 200, r
    return reply


class TestClient(unittest.TestCase):
    """One unit asked three questions, one of each kind, through a scripted wire."""

    @classmethod
    def setUpClass(cls):
        cls.tt = TempTask(rows=ticket_rows(300), task=ticket_task(), prompts={"three": THREE})
        cls.tt.ok("split")
        study = Study(cls.tt.dir)
        cls.ask = study.prompt(cls.tt.prompt("three")).asks(study.rows("train").rows[:1])[0]

    @classmethod
    def tearDownClass(cls):
        cls.tt.cleanup()

    def decide(self, *script, retries=3):
        """The reply of a client whose wire answers its n-th request with script[n](body)."""
        steps = iter(script)
        self.wire = Wire(lambda body: next(steps)(body))
        client = gateway.Client("typesafe-ai/jev", GatewaySettings(1, retries, 5.0), key="sk-test", transport=self.wire.transport)
        return client.decide(self.ask)

    def test_a_request_holds_the_values_as_json_and_the_questions_in_order(self):
        self.decide(lambda body: (200, recorded(body)))
        sent = self.wire.requests[0]
        self.assertEqual((sent.url, sent.method, sent.headers["Authorization"]), (gateway.URL, "POST", "Bearer sk-test"))
        self.assertEqual(sent.body, {
            "model": "typesafe-ai/jev",
            "input": [{"role": "user", "content": [{"type": "input_json", "data": dict(self.ask.data)}]}],
            "questions": [{"type": "choice", "name": "queue", "instructions": "Which queue?",
                           "choices": [{"value": "other"}, {"value": "billing"}, {"value": "shipping"}]},
                          {"type": "score", "name": "urgency", "instructions": "How urgent?",
                           "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}]},
                          {"type": "predicate", "name": "big", "instructions": "Is x above 1?"}]})

    def test_each_kind_is_read_with_its_probabilities_keyed_by_value_and_the_cost(self):
        reply = self.decide(lambda body: (200, recorded(body)))
        self.assertEqual(reply.answers, (Chose((0.2, 0.7, 0.1), 1), Scored((0.08, 0.54, 0.38), 1.3), Yes(0.95)))
        self.assertEqual((reply.cost_usd, reply.error, reply.stop), (0.0000096, "", False))

    def test_a_refused_question_is_refused_and_the_others_are_answered(self):
        reply = self.decide(changed(1, type="refusal"))
        self.assertEqual(reply.answers[1:], (Refused(), Yes(0.95)))
        self.assertIsInstance(reply.answers[0], Chose)

    def test_without_probabilities_only_the_point_is_kept(self):
        for missing in ({}, {"probabilities": {}}, {"probabilities": [{"value": "other", "probability": 1.0}]}):
            with self.subTest(missing=missing):
                reply = self.decide(changed(0, choice="shipping", **missing))
                self.assertEqual(reply.answers[0], Chose(None, 2))
        self.assertEqual(self.decide(changed(1, score=2)).answers[1], Scored(None, 2.0))

    def test_a_429_waits_as_long_as_retry_after_says_then_asks_again(self):
        now = 1_800_000_000.0
        for header, wait in (("7", 7.0), (formatdate(now + 30, usegmt=True), 30.0)):
            with self.subTest(header=header):
                limited = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_exceeded"}}
                reply = self.decide(lambda body: (429, limited, {"Retry-After": header}), lambda body: (200, recorded(body)))
                self.assertEqual(self.wire.waits, [wait])
                self.assertIsNotNone(reply.answers)

    def test_server_errors_back_off_and_the_retries_are_bounded(self):
        down = {"error": {"message": "unavailable", "type": "service_unavailable"}}
        reply = self.decide(*[lambda body: (503, down)] * 3, retries=2)
        self.assertEqual((len(self.wire.requests), self.wire.waits), (3, [1, 2]))
        self.assertEqual((reply.answers, reply.stop), (None, False))
        self.assertIn("HTTP 503", reply.error)

    def test_a_402_stops_at_once(self):
        broke = {"error": {"message": "no credit", "type": "quota_for_entity_exceeded"}}
        reply = self.decide(lambda body: (402, broke), lambda body: (200, recorded(body)))
        self.assertEqual((len(self.wire.requests), self.wire.waits, reply.stop), (1, [], True))
        self.assertIn("HTTP 402", reply.error)

    def test_answers_to_other_questions_stop_the_run_and_keep_the_cost(self):
        def short(body):
            r = recorded(body)
            r["answers"].pop()
            return 200, r
        for script, said in ((changed(2, name="small"), "'small'"), (short, "2 answers to 3 questions")):
            with self.subTest(said=said):
                reply = self.decide(script)
                self.assertEqual((reply.answers, reply.stop, reply.cost_usd), (None, True, 0.0000096))
                self.assertIn(said, reply.error)

    def test_a_malformed_answer_is_an_error_asked_again_next_run(self):
        for k, item in ((2, {"probability": "high"}), (0, {"choice": "sales"}), (1, {"score": 3.5})):
            with self.subTest(item=item):
                reply = self.decide(changed(k, **item))
                self.assertEqual((reply.answers, reply.stop), (None, False))
                self.assertIn("bad reply", reply.error)


SENTINEL = "sk-sentinel-0123456789abcdef"


def in_window(body):
    return "13:00" <= data_of(body)["time_utc"] <= "15:59"


class TestThroughTheCommandLine(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask()
        self.tt.ok("split")
        self.window = self.tt.prompt("window")
        env = mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": SENTINEL})
        env.start()
        os.environ.pop("PLAB_KEY_FILE", None)
        self.addCleanup(env.stop)

    def tearDown(self):
        self.tt.cleanup()

    def run_jev(self, answer, *more):
        wire = Wire(answer)
        with wire.patch():
            return self.tt.plab("run", self.window, "--decider", "jev", *more), wire

    def ledger(self):
        return self.tt.log("ledger", "lab")

    def test_the_cost_the_gateway_reports_goes_to_the_ledger_and_answers_are_asked_once(self):
        r, wire = self.run_jev(lambda body: (200, yes_reply(body, 0.9 if in_window(body) else 0.1, cost="0.0000125")))
        self.assertEqual(r.code, 0, r.err)
        calls = sum(line["calls"] for line in self.ledger())
        self.assertEqual(calls, len(wire.requests))
        self.assertAlmostEqual(sum(line["cost_usd"] for line in self.ledger()), 0.0000125 * calls, places=9)
        again, wire = self.run_jev(lambda body: (500, b""), "--json")
        self.assertEqual((json.loads(again.out)["calls"], wire.requests), (0, []))

    def test_a_402_stops_the_run_and_what_was_paid_for_is_kept(self):
        self.tt.write_task({**alert_task(), "gateway": {"workers": 1, "retries": 4, "timeout_s": 5}})
        count = itertools.count()
        r, wire = self.run_jev(lambda body: (200, yes_reply(body, 0.1)) if next(count) < 10 else (402, {"error": {"message": "no credit"}}))
        self.assertEqual(r.code, 2)
        self.assertIn("HTTP 402", r.err)
        self.assertEqual((len(wire.requests), wire.waits), (11, []))
        self.assertEqual([line["calls"] - line["errors"] for line in self.ledger()], [10])
        r, wire = self.run_jev(lambda body: (200, yes_reply(body, 0.1)), "--json")
        self.assertEqual(json.loads(r.out)["calls"], len(wire.requests))
        self.assertEqual(sum(line["calls"] - line["errors"] for line in self.ledger()), 10 + len(wire.requests))

    def test_the_key_is_in_no_output_store_ledger_or_log(self):
        """The wire echoes the authorization header in every error body, as a careless server might."""
        def careless(body):
            minute = int(data_of(body)["time_utc"][3:])
            if minute % 7 == 0:
                return 400, {"error": {"message": f"bad request with Authorization: Bearer {SENTINEL}"}}
            if minute % 7 == 1:
                return 200, b"not json " + SENTINEL.encode()
            return 200, yes_reply(body, 0.5)
        outputs = [self.run_jev(careless)[0], self.run_jev(lambda body: (402, {"error": {"message": f"key {SENTINEL} has no credit"}}))[0]]
        self.assertEqual([r.code for r in outputs], [2, 2])
        self.assertIn("[key]", outputs[1].err)
        written = [p for p in (self.tt.dir / ".plab").rglob("*") if p.is_file()]
        self.assertTrue(any("answers" in p.parts for p in written) and any(p.name == "ledger.jsonl" for p in written))
        for p in written:
            self.assertNotIn(SENTINEL.encode(), p.read_bytes(), p)
        for r in outputs:
            self.assertNotIn(SENTINEL, r.out + r.err)

    def test_the_key_can_come_from_the_file_plab_key_file_names(self):
        os.environ.pop("AI_GATEWAY_API_KEY")
        r, wire = self.run_jev(lambda body: (200, yes_reply(body, 0.1)))
        self.assertEqual((r.code, wire.requests), (2, []))
        self.assertIn("no key", r.err)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "key").write_text(SENTINEL + "\n", encoding="utf-8")
            os.environ["PLAB_KEY_FILE"] = str(Path(d) / "key")
            r, wire = self.run_jev(lambda body: (200, yes_reply(body, 0.1)))
        self.assertEqual(r.code, 0, r.err)
        self.assertEqual({q.headers["Authorization"] for q in wire.requests}, {f"Bearer {SENTINEL}"})


if __name__ == "__main__":
    unittest.main()
