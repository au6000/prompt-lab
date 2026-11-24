"""A small task in a temporary folder, `plab` run in this process, and a stand-in for the network
(no network, no key)."""
import contextlib
import email.message
import io
import json
import shutil
import tempfile
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from promptlab import gateway
from promptlab.cli import main

PREREG = "# Pre-registration (test)\n\nChoose by the rule in task.json.\n"

WINDOW = {"name": "window", "fields": ["time_utc"],
          "veto_rules": [{"text": "(a) Skip when time_utc is between 13:00 and 15:59.", "expr": "'13:00' <= time_utc <= '15:59'"}]}
TREND = {"name": "trend", "fields": ["x"], "veto_rules": [{"text": "(b) Skip when x is above 1.", "expr": "x > 1"}]}
WORDS = {"name": "words", "fields": ["x"], "veto_rules": ["(q) Skip when x looks unusually big."]}

ROUTE = {"name": "route", "fields": ["topic", "x"],
         "questions": [{"name": "queue", "type": "choice", "instructions": "Which queue should take this ticket?",
                        "choices": ["other", {"value": "billing", "description": "refunds and payments"}, "shipping"],
                        "expr": "topic"},
                       {"name": "big", "type": "predicate", "instructions": "Is x above 1?", "expr": "x > 1"}],
         "forecast": {"question": "queue", "classes": {"other": "other", "billing": "billing", "shipping": "shipping"}}}


def plab(*args):
    """Run `plab args` in this process. Returns its exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(args))
    return SimpleNamespace(code=code, out=out.getvalue(), err=err.getvalue())


def _times(rng, n):
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    span = (datetime(2026, 1, 1, tzinfo=timezone.utc) - start).total_seconds()
    return sorted(start + timedelta(seconds=float(x)) for x in rng.uniform(0, span, n))


def alert_rows(n=1600, seed=3):
    """Two classes with a value: 1 loses 50, 0 wins 2. Class 1 is more likely inside the 13:00-15:59
    window, and before 2022 also when x is above 1."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, t in enumerate(_times(rng, n)):
        hhmm = f"{t.hour:02d}:{t.minute:02d}"
        x = float(rng.normal(0, 1))
        p = 0.05 * (2.5 if "13:00" <= hhmm <= "15:59" else 1.0) * (2.0 if (x > 1 and t.year < 2022) else 1.0)
        y = int(rng.random() < p)
        rows.append({"id": f"r{k:05d}", "t": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "state": {"time_utc": hhmm, "x": round(x, 2), "noise": round(float(rng.random()), 3),
                               "maybe": None if rng.random() < 0.2 else round(float(rng.random()), 2),
                               "series": [round(float(v), 2) for v in rng.normal(0, 1, 12)], "flag": bool(rng.random() < 0.3)},
                     "label": y, "value": -50.0 if y else 2.0})
    return rows


def ticket_rows(n=1500, seed=5):
    """Three classes without a value. The state's topic names the queue four times in five."""
    rng = np.random.default_rng(seed)
    queues = ["other", "billing", "shipping"]
    rows = []
    for k, t in enumerate(_times(rng, n)):
        queue = queues[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]
        topic = queue if rng.random() < 0.8 else queues[int(rng.integers(3))]
        rows.append({"id": f"t{k:05d}", "t": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "state": {"topic": topic, "x": round(float(rng.normal(0, 1)), 2)}, "queue": queue})
    return rows


def alert_task(**over):
    task = {
        "format": 2,
        "name": "unit",
        "rows": {"value": "value"},
        "splits": {"train": ["2018-01-01", "2022-01-01"], "eval": ["2022-01-01", "2024-01-01"], "test": ["2024-01-01", "2026-01-01"]},
        "half_boundary": "2020-01-01",
        "train_sample": {"reference_per_other": 4, "seed": 1},
        "decision": {"skip_if_prob_at_least": 0.5},
        "selection": {"require": [{"metric": "eval.decision.gain_vs_all_in_per_1000", "op": ">", "value": 0}],
                      "rank_by": [{"metric": "eval.decision.gain_vs_all_in_per_1000", "order": "desc"}]},
        "budget": {"usd": 1.0, "eval_checks_per_agent": 2, "nominations_per_agent": 1},
        "decider": "fake",
        "metrics": {"cv_folds": 5, "cv_reps": 1, "mi_perms": 30, "null_reps": 6, "null_seed": 1},
    }
    return {**task, **over}


def ticket_task(**over):
    task = {
        "format": 2,
        "name": "tickets",
        "rows": {"outcome": "queue"},
        "outcome": {"classes": ["other", "billing", "shipping"]},
        "splits": {"train": ["2018-01-01", "2022-01-01"], "eval": ["2022-01-01", "2024-01-01"], "test": ["2024-01-01", "2026-01-01"]},
        "selection": {"require": [{"metric": "train.cv_gain_pct", "op": ">", "value": 0},
                                  {"metric": "eval.null_pct.cv_gain_pct", "op": ">=", "value": 90}],
                      "rank_by": [{"metric": "eval.cv_gain_pct", "order": "desc"}]},
        "budget": {"usd": 1.0, "eval_checks_per_agent": 2, "nominations_per_agent": 1},
        "decider": "fake",
        "metrics": {"cv_folds": 5, "cv_reps": 1, "mi_perms": 30, "null_reps": 20, "null_seed": 1},
    }
    return {**task, **over}


class TempTask:
    """A task folder with rows, task.json, prereg.md and prompts. plab() runs a command on it."""

    def __init__(self, rows=None, task=None, prereg=PREREG, prompts=None):
        self.dir = Path(tempfile.mkdtemp(prefix="plab_test_"))
        self.write_rows(alert_rows() if rows is None else rows)
        self.write_task(alert_task() if task is None else task)
        if prereg is not None:
            (self.dir / "prereg.md").write_text(prereg, encoding="utf-8")
        (self.dir / "prompts").mkdir()
        for name, prompt in ({"window": WINDOW, "trend": TREND, "words": WORDS} if prompts is None else prompts).items():
            self.write_prompt(name, prompt)

    def write_rows(self, rows):
        (self.dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def write_task(self, task):
        (self.dir / "task.json").write_text(json.dumps(task), encoding="utf-8")

    def task(self):
        return json.loads((self.dir / "task.json").read_text(encoding="utf-8"))

    def write_prompt(self, name, prompt):
        path = self.dir / "prompts" / f"{name}.json"
        path.write_text(json.dumps(prompt), encoding="utf-8")
        return str(path)

    def prompt(self, name):
        return str(self.dir / "prompts" / f"{name}.json")

    def plab(self, *args):
        return plab("-C", str(self.dir), *args)

    def ok(self, *args):
        """Run a command that must succeed. Returns its stdout."""
        r = self.plab(*args)
        if r.code != 0:
            raise AssertionError(f"plab {' '.join(args)} exited with {r.code}: {r.err}")
        return r.out

    def record(self, *args):
        """Run a command with --json that must succeed. Returns what it printed."""
        return json.loads(self.ok(*args, "--json"))

    def split_rows(self, split):
        text = (self.dir / ".plab" / "data" / f"{split}.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def log(self, kind, agent):
        path = self.dir / ".plab" / "agents" / agent / f"{kind}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class Wire:
    """The network as gateway.Client sees it. `answer(body)` gives (status, reply) or (status, reply,
    headers) for each request body; a reply that is not bytes is sent as JSON. Requests and waits
    are recorded."""

    def __init__(self, answer, now=1_800_000_000.0):
        self.answer, self.requests, self.waits = answer, [], []
        self.transport = gateway.Transport(self.open, self.waits.append, lambda: now)

    def open(self, request, timeout):
        body = json.loads(request.data)
        self.requests.append(SimpleNamespace(url=request.full_url, method=request.get_method(),
                                             headers=dict(request.header_items()), body=body))
        status, reply, *headers = self.answer(body)
        data = reply if isinstance(reply, bytes) else json.dumps(reply).encode("utf-8")
        if status == 200:
            return io.BytesIO(data)
        message = email.message.Message()
        for k, v in (headers[0] if headers else {}).items():
            message[k] = v
        raise urllib.error.HTTPError(request.full_url, status, "error", message, io.BytesIO(data))

    def patch(self):
        """While active, every gateway decider calls through this wire."""
        return mock.patch.object(gateway, "NETWORK", self.transport)


def yes_reply(body, p, cost="0.00002"):
    """A Decisions API reply that answers every question of a request, all predicates, with p."""
    return {"model": body["model"], "answers": [{"type": "predicate", "name": q["name"], "probability": p} for q in body["questions"]],
            "usage": {"input_tokens": 96, "output_tokens": 0, "total_tokens": 96},
            "provider_metadata": {"gateway": {"cost": cost, "generationId": "gen_01"}}}


def data_of(body):
    """The unit's values in a request."""
    return body["input"][0]["content"][0]["data"]
