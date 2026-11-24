"""The Vercel AI Gateway Decisions API (POST /v1/decisions), with the standard library only.

The wire boundary: an Ask goes out as a request and the response comes back as a Reply. The only
wire shapes outside this file are the questions' own (questions.wire, questions.parse_answer). The key
comes only from AI_GATEWAY_API_KEY or the file PLAB_KEY_FILE names. A Client keeps it in a closure,
not in an attribute, and every error text has the key replaced, so it reaches no output, store or
ledger.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .answers import Reply
from .layout import PlabError
from .prompt import Ask
from .questions import Answer, parse_answer, wire
from .task import GatewaySettings

URL = "https://ai-gateway.vercel.sh/v1/decisions"
RETRY = (429, 500, 502, 503, 504, 529)   # wait, then ask again
STOP = (401, 402, 403)                   # key refused, no credit left (402 is never retried), plan: no call can succeed


class Mismatch(ValueError):
    """A reply whose answers are not to the questions asked: another count, name or type."""


@dataclass(frozen=True)
class Transport:
    """Everything the client touches outside the process. Tests pass their own."""
    open: Callable[..., Any]          # as urllib.request.urlopen(request, timeout=...)
    sleep: Callable[[float], None]
    clock: Callable[[], float]


NETWORK = Transport(urllib.request.urlopen, time.sleep, time.time)


def load_key() -> str:
    key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("PLAB_KEY_FILE")
    if not path:
        raise PlabError("no key: set AI_GATEWAY_API_KEY, or PLAB_KEY_FILE to the path of a file that holds the key")
    if not Path(path).is_file():
        raise PlabError("PLAB_KEY_FILE is set but does not name a readable file")
    key = Path(path).read_text(encoding="utf-8").strip()
    if not key:
        raise PlabError("the file that PLAB_KEY_FILE names is empty")
    return key


def sent(ask: Ask) -> dict[str, Any]:
    """An ask as a request, without the model: the unit's values as one JSON part, then the questions."""
    return {"input": [{"role": "user", "content": [{"type": "input_json", "data": ask.data}]}],
            "questions": [wire(q) for q in ask.questions]}


def cost_usd(body: Any) -> float:
    """provider_metadata.gateway.cost, US dollars as a decimal string; 0 when the reply gives none."""
    gateway = ((body.get("provider_metadata") or {}).get("gateway") or {}) if isinstance(body, dict) else {}
    cost = gateway.get("cost")
    return 0.0 if cost in (None, "") else float(cost)


def parse_reply(ask: Ask, body: Any) -> tuple[Answer, ...]:
    """The answers of a reply, matched to the questions by position and by name. Mismatch when they
    answer other questions; ValueError, KeyError or TypeError when an answer is malformed."""
    items = body["answers"]
    if not isinstance(items, list) or len(items) != len(ask.questions):
        count = len(items) if isinstance(items, list) else "no list of"
        raise Mismatch(f"{count} answers to {len(ask.questions)} questions")
    for k, (q, item) in enumerate(zip(ask.questions, items), 1):
        if not isinstance(item, dict) or item.get("name") != q.name or item.get("type") not in (q.kind, "refusal"):
            got = (item.get("type"), item.get("name")) if isinstance(item, dict) else item
            raise Mismatch(f"answer {k} is {got!r}; question {k} is the {q.kind} {q.name!r}")
    return tuple(parse_answer(q, item) for q, item in zip(ask.questions, items))


def retry_after_s(header: str | None, now: float) -> float | None:
    """The wait a Retry-After header asks for, given in seconds or as an HTTP date; None without one."""
    if not header:
        return None
    try:
        wait = float(header)
    except ValueError:
        try:
            wait = parsedate_to_datetime(header).timestamp() - now
        except (TypeError, ValueError):
            return None
    return max(0.0, wait) if math.isfinite(wait) else None


def _error_text(e: urllib.error.HTTPError) -> str:
    try:
        return f"HTTP {e.code} {e.read().decode('utf-8', 'replace')}"
    except (OSError, http.client.HTTPException):
        return f"HTTP {e.code}"


class Client:
    """One model and one key. decide() makes at most settings.retries + 1 attempts: a RETRY status or
    a network error waits (as Retry-After says, else 2**attempt seconds) and asks again; a STOP status
    is a stop reply at once; any other failure is an error reply, asked again on the next run."""

    def __init__(self, model: str, settings: GatewaySettings, *, key: str | None = None, transport: Transport | None = None) -> None:
        self.model, self.settings, self.transport = model, settings, transport or NETWORK
        key = load_key() if key is None else key
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self._request = lambda data: urllib.request.Request(URL, data=data, method="POST", headers=headers)
        self._redact = lambda text: (text.replace(key, "[key]") if key else text)[:300]

    def decide(self, ask: Ask) -> Reply:
        data = json.dumps({"model": self.model, **sent(ask)}).encode("utf-8")
        attempt = 0
        while True:
            reply, wait = self._attempt(ask, data, attempt)
            if wait is None or attempt == self.settings.retries:
                return reply
            self.transport.sleep(wait)
            attempt += 1

    def _attempt(self, ask: Ask, data: bytes, attempt: int) -> tuple[Reply, float | None]:
        """One call: its reply, and the seconds to wait before asking again when it may be retried."""
        t = self.transport
        try:
            with t.open(self._request(data), timeout=self.settings.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as e:
            reply = Reply(None, error=self._redact(_error_text(e)), stop=e.code in STOP)
            if e.code not in RETRY:
                return reply, None
            wait = retry_after_s(e.headers.get("Retry-After"), t.clock())
            return reply, 2 ** attempt if wait is None else wait
        except (OSError, http.client.HTTPException) as e:
            return Reply(None, error=self._redact(f"network: {e!r}")), 2 ** attempt
        return self._reply(ask, raw), None

    def _reply(self, ask: Ask, raw: bytes) -> Reply:
        cost = 0.0
        try:
            body = json.loads(raw)
            cost = cost_usd(body)
            return Reply(parse_reply(ask, body), cost_usd=cost)
        except Mismatch as e:
            return Reply(None, error=self._redact(f"the reply answers other questions: {e}"), cost_usd=cost, stop=True)
        except (KeyError, TypeError, ValueError) as e:
            return Reply(None, error=self._redact(f"bad reply: {e!r}"), cost_usd=cost)

    def decide_many(self, asks: Sequence[Ask]) -> list[Reply]:
        """Replies in ask order, settings.workers calls at a time. Once a reply stops the run, the asks
        not yet sent are not sent."""
        stopped = threading.Event()

        def one(ask: Ask) -> Reply:
            if stopped.is_set():
                return Reply(None, error="not asked: an earlier call stopped the run")
            reply = self.decide(ask)
            if reply.stop:
                stopped.set()
            return reply
        with ThreadPoolExecutor(self.settings.workers) as pool:
            return list(pool.map(one, asks))
