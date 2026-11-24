"""The task folder on disk: every path the harness reads or writes, the file primitives, and the
refusal type. No other module builds a path under `.plab/` (migrate.py knows where version 0.1
kept its files).

    .plab/data/<split>.jsonl               the rows of one split, in time order
    .plab/answers/<shelf>/meta.json        what one shelf of stored answers was asked
    .plab/answers/<shelf>/<agent>.jsonl    the answers that agent fetched
    .plab/nulls/<split>.<signature>.json   zero points
    .plab/agents/<agent>/<kind>.jsonl      runs, checks, nominations, agree, ledger, surveys
    .plab/prereg.lock, prereg.frozen.md, selected.json, test.lock, test_result.json

Logs are per agent. An append on Windows is a seek and then a write, so two agents appending to
one file can overwrite each other; with a file each they cannot, and readers merge.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NewType

LogKind = Literal["runs", "checks", "nominations", "agree", "ledger", "surveys"]
AgentName = NewType("AgentName", str)

#: The names the harness itself logs under. `--agent` refuses them.
SELECT = AgentName("select")
TEST = AgentName("test")
MIGRATE = AgentName("migrate")   # what `plab migrate` carried over from version 0.1: answers and the ledger


class PlabError(Exception):
    """A refusal or a user error. The CLI prints the message and exits with code 2."""


def agent_name(raw: str | None) -> AgentName:
    """The name given with --agent. It becomes a folder name, so it starts with a letter or digit."""
    if not raw:
        raise PlabError("give --agent NAME (the name of the searching agent or person)")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}", raw):
        raise PlabError("--agent may use letters, digits, _ . - (1 to 40 characters, starting with a letter or digit)")
    if raw.lower() in (SELECT, TEST, MIGRATE):
        raise PlabError(f"--agent {raw} is reserved for the harness")
    return AgentName(raw)


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def plab(self) -> Path:
        return self.root / ".plab"

    @property
    def task_json(self) -> Path:
        return self.root / "task.json"

    @property
    def prereg_md(self) -> Path:
        return self.root / "prereg.md"

    def split_file(self, split: str) -> Path:
        return self.plab / "data" / f"{split}.jsonl"

    def shelf(self, key: str) -> Path:
        return self.plab / "answers" / key

    def null_file(self, split: str, signature: str) -> Path:
        return self.plab / "nulls" / f"{split}.{signature}.json"

    def null_files(self, split: str) -> list[Path]:
        return sorted((self.plab / "nulls").glob(f"{split}.*.json"))

    @property
    def prereg_lock(self) -> Path:
        return self.plab / "prereg.lock"

    @property
    def prereg_frozen(self) -> Path:
        return self.plab / "prereg.frozen.md"

    @property
    def selected(self) -> Path:
        return self.plab / "selected.json"

    @property
    def test_lock(self) -> Path:
        return self.plab / "test.lock"

    @property
    def test_result(self) -> Path:
        return self.plab / "test_result.json"

    def check_lock(self, agent: AgentName) -> Path:
        return self.plab / "agents" / agent / "check.lock"

    def log(self, agent: AgentName, kind: LogKind) -> Path:
        return self.plab / "agents" / agent / f"{kind}.jsonl"

    def append_log(self, agent: AgentName, kind: LogKind, record: dict[str, Any]) -> dict[str, Any]:
        """Append a record to an agent's log, stamped with the time and the agent. Returns the line."""
        line = {"time": now(), "agent": agent, **record}
        append_jsonl(self.log(agent, kind), [line])
        return line

    def read_log(self, kind: LogKind, agent: AgentName | None = None) -> list[dict[str, Any]]:
        """The lines of one kind: one agent's in the order written, or every agent's merged by time."""
        if agent is not None:
            return read_jsonl(self.log(agent, kind))
        lines = [(r.get("time", ""), f.parent.name, k, r)
                 for f in sorted((self.plab / "agents").glob(f"*/{kind}.jsonl")) for k, r in enumerate(read_jsonl(f))]
        return [r for *_, r in sorted(lines, key=lambda x: x[:3])]


def now() -> str:
    """The current time in UTC, e.g. 2026-11-19T07:51:00Z."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """A torn last line (an interrupted append) is skipped; any other bad line refuses."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if k < len(lines) - 1:
                raise PlabError(f"{path}: line {k + 1} is not valid JSON")
    return out


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write beside the target, then replace it: a crash leaves the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def short_hash(obj: Any) -> str:
    return hashlib.sha1(canonical(obj).encode("utf-8")).hexdigest()[:12]
