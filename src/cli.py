"""plab: the command line. COMMANDS is the registry; the help, the argument parser and the dispatch
are made from it. A handler is a thin shell over study, protocol or explore and prints through
show.py. Results go to stdout, progress to stderr. A refusal (PlabError) exits with code 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import __version__, explore, nulls, protocol, show, survey
from .answers import Spend
from .layout import SELECT, TEST, PlabError, agent_name
from .migrate import migrate
from .prompt import lint
from .rows import split_rows
from .study import Study

Args = argparse.Namespace
Say = Callable[[str], None]
DEFAULT_AGENT = "lab"   # who training runs are logged under when --agent is not given


def _init(a: Args, say: Say) -> None:
    made = explore.init(Path(a.dir))
    say(f"made {made}: task.json, prereg.md, prompts/example.json, .gitignore")
    say("next: put rows.jsonl there (one row per line: {id, t, state: {...}, label, value}), edit task.json, then `plab split`")


def _migrate(a: Args, say: Say) -> None:
    for line in migrate(Path(a.task), write=a.write, unfreeze=a.unfreeze):
        say(line)


def _split(study: Study, a: Args, say: Say) -> None:
    protocol.permit(study.layout, "split")
    show.split(split_rows(study.task, study.layout, force=a.force), say)


def _fields(study: Study, a: Args, say: Say) -> None:
    show.fields(explore.fields(study, with_auc=a.auc), say)


def _lint(study: Study, a: Args, say: Say) -> None:
    prompt = study.prompt(a.prompt)
    show.lint(prompt, *lint(prompt, study.rows("train").rows, study.all_fields()), say)


def _run(study: Study, a: Args, say: Say) -> None:
    """The training split with one decider, or with several to compare decision models."""
    agent = agent_name(a.agent or DEFAULT_AGENT)
    asker = study.prompt(a.prompt)
    candidates = [study.candidate(asker, name) for name in a.decider or [None]]
    spend = Spend("run", agent, a.cache_only, a.yes)
    records = [study.layout.append_log(agent, "runs", study.measure(c, study.design("train"), spend).record) for c in candidates]
    if a.json:
        say(_json(records[0] if len(records) == 1 else records))
    elif len(records) == 1:
        show.result(records[0], f"TRAINING: {asker.name}  [{asker.prompt_id}]", say)
    else:
        show.comparison(records, say)


def _agree(study: Study, a: Args, say: Say) -> None:
    spend = Spend("agree", agent_name(a.agent or DEFAULT_AGENT), a.cache_only, a.yes)
    cand = study.candidate(study.prompt(a.prompt), a.decider)
    show.agree(explore.agree(study, cand, spend, per_rule=a.per_rule, limit=a.limit), say)


def _drift(study: Study, a: Args, say: Say) -> None:
    cand = study.candidate(study.prompt(a.prompt), a.decider)
    show.drift(explore.drift(study, cand, Spend("drift", agent_name(DEFAULT_AGENT), cache_only=True), by=a.by), say)


def _survey(study: Study, a: Args, say: Say) -> None:
    asker = study.prompt(a.prompt) if a.prompt else None
    found = survey.survey(study, asker, by=a.by, gap=a.gap, every_split=a.splits == "all", agent=agent_name(a.agent or DEFAULT_AGENT))
    if a.json:
        say(_json(found))
    else:
        show.survey(found, say)


def _null(study: Study, a: Args, say: Say) -> None:
    lock = protocol.enter(study, "null") if a.split == "eval" else None
    signature = nulls.Signature.of(study.prompt(a.prompt), study.task.outcome.K)
    show.zero_point(nulls.get(study.task, study.layout, study.design(a.split, lock), signature, force=a.force, note=study.note), say)


def _board(study: Study, a: Args, say: Say) -> None:
    agent = agent_name(a.agent) if a.agent else None
    show.board(explore.board(study, split=a.split, agent=agent, show_all=a.all), split=a.split, top=a.top, say=say)


def _cost(study: Study, a: Args, say: Say) -> None:
    show.cost(explore.cost(study), say)


def _brief(study: Study, a: Args, say: Say) -> None:
    say(explore.brief(study, agent_name(a.agent), a.direction))


def _prereg(study: Study, a: Args, say: Say) -> None:
    if a.action == "status":
        say(protocol.lock_state(study))
        return
    frozen = protocol.freeze(study, force=a.force)
    data = frozen["data_sha256"]
    say(f"frozen: prereg.md {frozen['prereg_sha256'][:12]}, task settings {frozen['task_locked_sha256'][:12]}, "
        f"data train {data['train'][:12]} eval {data['eval'][:12]}" + (f" test {data['test'][:12]}" if data["test"] else ""))


def _check(study: Study, a: Args, say: Say) -> None:
    agent = agent_name(a.agent)
    cand = study.candidate(study.prompt(a.prompt), a.decider)
    r = protocol.check(study, cand, Spend("check", agent, a.cache_only, a.yes))
    if a.json:
        say(_json(r))
        return
    show.result(r, f"EVALUATION CHECK {r['check_no']}/{r['cap']} (agent {agent}): {r['name']}  [{r['prompt_id']}]", say)
    trained = [t for t in study.layout.read_log("runs") if (t["prompt_id"], t["decider"]) == (r["prompt_id"], r["decider"])]
    if trained:
        skip = (f"  skip {100 * trained[-1]['decision']['skip_frac']:.1f}% (evaluation {100 * r['decision']['skip_frac']:.1f}%)"
                if "decision" in trained[-1] and "decision" in r else "")
        say(f"  the last training run of this candidate: cv_gain {show.f(trained[-1]['cv_gain_pct'], '.2f')}%{skip}")
    say(f"evaluation checks used by {agent}: {r['check_no']}/{r['cap']}")


def _nominate(study: Study, a: Args, say: Say) -> None:
    agent = agent_name(a.agent)
    line = protocol.nominate(study, study.candidate(study.prompt(a.prompt), a.decider), agent)
    say(f"nominated {line['name']} [{line['prompt_id']}] with decider {line['decider']} "
        f"({len(study.layout.read_log('nominations', agent))}/{study.task.budget.nominations_per_agent} for {agent})")


def _select(study: Study, a: Args, say: Say) -> None:
    show.selection(protocol.select(study, Spend("select", SELECT, a.cache_only, a.yes)), say)


def _test(study: Study, a: Args, say: Say) -> None:
    if not a.final:
        raise PlabError("the test set is used once, for the selected candidate only: run `plab test --final`")
    show.test(protocol.test_final(study, Spend("test", TEST, over_budget_ok=a.yes), resume=a.resume), say)


def _report(study: Study, a: Args, say: Say) -> None:
    found = protocol.report(study)
    if a.json:
        say(_json(found))
    else:
        show.report(found, say)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def _on_study(handler: Callable[[Study, Args, Say], None]) -> Callable[[Args, Say], None]:
    def run(a: Args, say: Say) -> None:
        handler(Study(a.task, lambda text: print(text, file=sys.stderr, flush=True)), a, say)
    return run


def _arg(*names: str, **spec: Any) -> Callable[[argparse.ArgumentParser], Any]:
    return lambda parser: parser.add_argument(*names, **spec)


PROMPT = _arg("prompt", help="a prompt file")
AGENT = _arg("--agent", help="the name of the searching agent or person")
DECIDER = _arg("--decider", help="a decider of task.json (default: the task's `decider`)")
CACHE_ONLY = _arg("--cache-only", action="store_true", help="never call a decider; refuse when an answer is not stored")
YES = _arg("--yes", action="store_true", help="go ahead even if the estimated cost exceeds the remaining budget")
JSON = _arg("--json", action="store_true", help="print the full result as JSON")


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    run: Callable[[Args, Say], None]
    args: tuple[Callable[[argparse.ArgumentParser], Any], ...] = ()


COMMANDS = (
    Command("init", "make a task folder (task.json, prereg.md, prompts/)", _init, (_arg("dir"),)),
    Command("migrate", "convert a prompt-lab 0.1 task folder; prints the plan, --write does it", _migrate,
            (_arg("--write", action="store_true"),
             _arg("--unfreeze", action="store_true", help="set a 0.1 lock, its checks, nominations and selection aside"))),
    Command("split", "the rows file -> .plab/data/{train,eval,test}.jsonl by time", _on_study(_split),
            (_arg("--force", action="store_true"),)),
    Command("fields", "the fields of the state on the training split", _on_study(_fields), (_arg("--auc", action="store_true"),)),
    Command("lint", "wordings that lower the agreement of model and machine; what a prompt leaves out", _on_study(_lint), (PROMPT,)),
    Command("run", "the training split; --decider machine is free, several --decider compare models", _on_study(_run),
            (PROMPT, AGENT, _arg("--decider", action="append", help="a decider of task.json; repeat it to compare deciders"),
             CACHE_ONLY, YES, JSON)),
    Command("agree", "the training split: does the decider answer like the machine?", _on_study(_agree),
            (PROMPT, AGENT, DECIDER, _arg("--per-rule", action="store_true"), _arg("--limit", type=int), CACHE_ONLY, YES)),
    Command("drift", "the training split by period: answers by option, or rule hits and value", _on_study(_drift),
            (PROMPT, DECIDER, _arg("--by", choices=["quarter", "half", "year"], default="quarter"))),
    Command("survey", "does each factor's relation to the outcome hold across periods? (training split; free)", _on_study(_survey),
            (_arg("prompt", nargs="?", help="survey this prompt's rules and questions, as the machine answers them, instead of the state fields"),
             _arg("--by", choices=["quarter", "half", "year"], default="half"),
             _arg("--gap", type=int, default=0, help="whole months before an outcome is known (0)"),
             _arg("--splits", choices=["train", "all"], default="train",
                  help="all: every split of a closed study, exploratory and logged"),
             AGENT, JSON)),
    Command("null", "the zero point: the yardsticks on random answers shaped like the prompt's", _on_study(_null),
            (PROMPT, _arg("--split", choices=["train", "eval"], default="train"), _arg("--force", action="store_true"))),
    Command("board", "training runs, or an agent's own evaluation checks", _on_study(_board),
            (_arg("--split", choices=["train", "eval"], default="train"), AGENT, _arg("--all", action="store_true"),
             _arg("--top", type=int, default=25))),
    Command("cost", "what was spent, and what is left of the budget", _on_study(_cost)),
    Command("brief", "the brief for a searching agent", _on_study(_brief), (AGENT, _arg("--direction", default="(free)"))),
    Command("prereg", "freeze | status: fix prereg.md, the settings and the splits before any evaluation look", _on_study(_prereg),
            (_arg("action", choices=["freeze", "status"]), _arg("--force", action="store_true"))),
    Command("check", "ONE look at the evaluation split (capped per agent, logged)", _on_study(_check),
            (PROMPT, AGENT, DECIDER, CACHE_ONLY, YES, JSON)),
    Command("nominate", "put a prompt and a decider forward for the selection", _on_study(_nominate), (PROMPT, AGENT, DECIDER)),
    Command("select", "apply the pre-registered selection rule to the nominations", _on_study(_select), (CACHE_ONLY, YES)),
    Command("test", "--final: the selected candidate on the test split, once", _on_study(_test),
            (_arg("--final", action="store_true"), _arg("--resume", action="store_true"), YES)),
    Command("report", "the study so far: looks, nominations, selection, test, cost", _on_study(_report), (JSON,)),
)


def build_parser() -> argparse.ArgumentParser:
    overview = ("Measure how much a prompt lets a decision model tell about a known outcome, without overfitting.\n\n"
                + "\n".join(f"  {c.name:9} {c.help}" for c in COMMANDS))
    parser = argparse.ArgumentParser(prog="plab", description=overview, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-C", "--task", default=".", help="the task folder (default: the current folder)")
    parser.add_argument("--version", action="version", version=f"prompt-lab {__version__}")
    commands = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    for c in COMMANDS:
        sub = commands.add_parser(c.name, description=c.help)
        sub.set_defaults(run=c.run)
        for add in c.args:
            add(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = build_parser()
    a = parser.parse_args(argv)
    if not a.cmd:
        parser.print_help()
        return 0
    try:
        a.run(a, print)
    except PlabError as e:
        print(f"plab: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("plab: interrupted (the answers fetched so far are stored)", file=sys.stderr)
        return 130
    return 0
