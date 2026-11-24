"""Run the whole study on the synthetic data, without a key (the machine and the fake decider).

    python examples/synthetic/walkthrough.py [--reset]

--reset deletes examples/synthetic/.plab first (the study log), so the walkthrough can be repeated.
Each step prints the command it runs. Steps that the harness is meant to refuse are marked
"(expected refusal)" and must exit with code 2.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def plab(*args, expect=0):
    cmd = [sys.executable, "-m", "promptlab", "-C", str(HERE), *args]
    print(f"\n$ plab {' '.join(args)}" + ("   (expected refusal)" if expect == 2 else ""), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != expect:
        raise SystemExit(f"step failed: exit code {r.returncode}, expected {expect}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset and (HERE / ".plab").exists():
        shutil.rmtree(HERE / ".plab")
    if not (HERE / "rows.jsonl").exists():
        subprocess.run([sys.executable, str(HERE / "make_data.py")], check=True)
    p = lambda n: str(Path("prompts") / f"{n}.json")
    machine = ("--decider", "machine", "--agent", "demo")
    plab("split")
    plab("fields")
    plab("survey")
    plab("survey", p("ab_both"))
    plab("lint", p("lint_demo"))
    plab("run", p("a_window"), *machine)
    plab("run", p("b_trend_abs"), *machine)
    plab("run", p("ab_both"), *machine)
    plab("drift", p("b_trend_abs"), "--by", "year", "--decider", "machine")
    plab("run", p("words_quiet"), "--agent", "demo")
    plab("agree", p("ab_both"), "--agent", "demo", "--decider", "fake", "--per-rule")
    plab("check", p("a_window"), *machine, expect=2)
    plab("survey", p("ab_both"), "--splits", "all", expect=2)
    plab("prereg", "freeze")
    for n in ("b_trend_abs", "b_trend_rel", "a_window", "ab_both"):
        plab("check", p(n), *machine)
    plab("check", p("words_quiet"), "--agent", "demo", expect=2)
    plab("board", "--split", "eval", "--agent", "demo")
    plab("nominate", p("a_window"), *machine)
    plab("nominate", p("ab_both"), *machine)
    plab("select")
    plab("test", expect=2)
    plab("test", "--final")
    plab("test", "--final", expect=2)
    plab("survey", p("ab_both"), "--splits", "all", "--agent", "demo")
    plab("report")

if __name__ == "__main__":
    main()
