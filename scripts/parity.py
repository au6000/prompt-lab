"""Capture every number a study writes, so a rewrite can be checked against the version before it.

    python scripts/parity.py capture TASK_DIR OUT.json [--clear FILE ...] --run "CMD" [--run "CMD" ...]
    python scripts/parity.py compare BEFORE.json AFTER.json [--allow-new] [--renames FILE]

capture copies TASK_DIR (with its .plab/) to a temporary folder, deletes the .plab logs named by
--clear, runs each CMD in that copy, and writes every number
found in .plab/*.json and .plab/*.jsonl (the stored answers excepted) as a flat map
"file|agent|name|split|form#k|field.path" -> value. In CMD, {py} is this interpreter and {dir} the copy.

compare exits 1 when a key of BEFORE is missing from AFTER or holds a different value, and when AFTER
has keys BEFORE lacks (unless --allow-new). --renames names a JSON list of {"old", "new", "why"}:
each "old" is a regular expression rewritten to "new" in BEFORE's keys, in order, for what a later
version moved or renamed on purpose. scripts/renames_since_0.1.json is that list for a BEFORE
captured from version 0.1.0.

The commands import promptlab from the interpreter's environment, so install this checkout first
(`pip install -e .`); the package lives in src/, which is not on the import path by itself.
"""
import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ID_FIELDS = ("agent", "name", "split", "form")


def leaves(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}[{i}]")
    elif value is None or isinstance(value, (bool, int, float)):
        yield path, value


def flatten(plab: Path) -> dict:
    out, seen = {}, {}
    stores = {"cache", "answers"}
    for f in sorted(p for p in plab.rglob("*") if p.suffix in (".json", ".jsonl") and not stores & set(p.parts)):
        rel = f.relative_to(plab).as_posix()
        text = f.read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()] if f.suffix == ".jsonl" else [json.loads(text)]
        for rec in records:
            ident = "|".join(str(rec.get(k, "")) for k in ID_FIELDS) if isinstance(rec, dict) else ""
            base = f"{rel}|{ident}"
            seen[base] = seen.get(base, 0) + 1
            for path, v in leaves(rec):
                out[f"{base}#{seen[base]}|{path}"] = v
    return out


def capture(a):
    with tempfile.TemporaryDirectory(prefix="parity-") as tmp:
        work = Path(tmp) / "task"
        shutil.copytree(a.task_dir, work, ignore=shutil.ignore_patterns("__pycache__"))
        for name in a.clear:
            (work / ".plab" / name).unlink(missing_ok=True)
        for cmd in a.run:
            argv = [part.format(py=sys.executable, dir=str(work)) for part in shlex.split(cmd, posix=True)]
            r = subprocess.run(argv, cwd=work, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                sys.exit(f"step failed ({r.returncode}): {cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        flat = flatten(work / ".plab")
    Path(a.out).write_text(json.dumps(flat, indent=1, sort_keys=True), encoding="utf-8")
    print(f"captured {len(flat)} numbers -> {a.out}")


def compare(a):
    before = json.loads(Path(a.before).read_text(encoding="utf-8"))
    after = json.loads(Path(a.after).read_text(encoding="utf-8"))
    for rename in json.loads(Path(a.renames).read_text(encoding="utf-8")) if a.renames else []:
        before = {re.sub(rename["old"], rename["new"], k): v for k, v in before.items()}
    missing = sorted(set(before) - set(after))
    new = sorted(set(after) - set(before))
    differ = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    for label, keys, show in (("missing in after", missing, lambda k: before[k]),
                              ("different", differ, lambda k: f"{before[k]} -> {after[k]}"),
                              ("new in after", new, lambda k: after[k])):
        for k in keys[:40]:
            print(f"{label}: {k} = {show(k)}")
        if len(keys) > 40:
            print(f"{label}: ... {len(keys) - 40} more")
    ok = not missing and not differ and (a.allow_new or not new)
    print(f"{'PARITY' if ok else 'MISMATCH'}: {len(before)} before, {len(after)} after, "
          f"{len(missing)} missing, {len(differ)} different, {len(new)} new")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("task_dir")
    c.add_argument("out")
    c.add_argument("--clear", nargs="*", default=[])
    c.add_argument("--run", action="append", required=True)
    m = sub.add_parser("compare")
    m.add_argument("before")
    m.add_argument("after")
    m.add_argument("--allow-new", action="store_true")
    m.add_argument("--renames")
    a = ap.parse_args()
    capture(a) if a.cmd == "capture" else compare(a)


if __name__ == "__main__":
    main()
