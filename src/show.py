"""Printing. Everything here formats what another module found; nothing is measured here."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

Say = Callable[[str], None]

SIGN_NOTE = ("cv_gain and MI do not see which way an answer points (a reversed answer scores the same). transfer_gain, asis_gain, "
             "AUC and the decision score it in a direction set beforehand: the forecast's, or one learned on other units.")


def f(x: Any, spec: str = ".3f") -> str:
    return "-" if x is None else format(x, spec)


def sign(x: float | None, spec: str = ".1f") -> str:
    return "-" if x is None else ("+" if x > 0 else "") + format(x, spec)


def _counts(counts: Mapping[str, int], *, shares: bool = False) -> str:
    """shares only for a whole split: a down-sampled design does not have the population's shares."""
    total = sum(counts.values())
    return ", ".join(f"{name}: {n}" + (f" ({100 * n / total:.2f}%)" if shares else "") for name, n in counts.items())


def split(found: Mapping[str, Any], say: Say) -> None:
    train = found["train"]
    for name, n in found["rows"].items():
        detail = ""
        if name == "train" and n:
            value = f", take-everything value per 1,000 {sign(train['all_in_value_per_1000'])}" if train["all_in_value_per_1000"] is not None else ""
            detail = f", classes {_counts(train['class_counts'], shares=True)}{value}, {train['first']} .. {train['last']}"
        say(f"{name:5}: {n} rows{detail}")
    say(f"dropped (outside every range): {found['dropped_outside_ranges']}")
    say("searchers see the training split only; the evaluation split goes through `plab check`, the test split through `plab test --final`")


def fields(found: Mapping[str, Any], say: Say) -> None:
    value = found["all_in_value_per_1000"]
    say(f"training split: {found['rows']} rows, classes {_counts(found['class_counts'], shares=True)}"
        + (f", take-everything value per 1,000 rows {sign(value)}" if value is not None else ""))
    even = found["breakeven"]
    if even:
        say(f"break-even: with mean values {even['mean_value_pos']} (positive) and {even['mean_value_neg']} (negative), the rows you take "
            f"must be below {even['breakeven_pos_pct']}% positive (above {even['breakeven_neg_pct']}% negative)")
    say(f"class meaning: {found['meaning']}")
    say("")
    if any("auc" in r for r in found["fields"]):
        say("single-field AUC on the (weighted) training design; > 0.5 means higher values go with the positive class. Training split only.")
    for r in found["fields"]:
        auc = f"   AUC {r['auc']:.3f} (halves {f(r['auc_h1'])} / {f(r['auc_h2'])})" if r.get("auc") is not None else ""
        say(f"  {r['field'][:60]:60} {r['type'][:12]:12} filled {r['filled_pct']:5.1f}%  {r['summary'] or ''}{auc}")


def lint(prompt: Any, warnings: Sequence[str], notes: Sequence[str], say: Say) -> None:
    ready = all(q.machine_ready for q in prompt.questions)
    say(f"{prompt.name} [{prompt.prompt_id}]: {len(prompt.questions)} questions, {len(prompt.rules)} rules, {len(prompt.fields)} fields, "
        + ("the machine can answer it (free: `plab run PROMPT --decider machine`)" if ready else "the machine cannot answer all of it"))
    for text in warnings:
        say(f"  WARN  {text}")
    for text in notes:
        say(f"  note  {text}")
    if not warnings:
        say("  no warnings")


def result(r: Mapping[str, Any], title: str, say: Say) -> None:
    """One measurement. Lines for yardsticks the prompt or the task cannot have are left out."""
    say(title)
    say(f"  split {r['split']}  decider {r['decider']} ({r['model']})  units {r['n']}  classes {_counts(r['class_counts'])}"
        + (f"  refused {r['refused']} (measured as the uniform answer)" if r["refused"] else "")
        + (f"  (missing answers {r['missing']})" if r["missing"] else ""))
    if r.get("answers_via"):
        say(f"  answers from two endpoints, measured as one: {', '.join(f'{n} via {e}' for e, n in r['answers_via'].items())}")
    zero = r["null_pct"]
    ref = zero["ref"]["cv_gain_pct"] or {}
    say(f"  cv_gain {f(r['cv_gain_pct'], '.2f')}%: at the {f(zero['cv_gain_pct'], '.1f')}th percentile of {zero['kind']} random answers "
        f"(random: median {f(ref.get('median'), '.2f')}%, 95th {f(ref.get('p95'), '.2f')}%)")
    if "transfer_gain_pct" in r:
        trained = r["trained_on"]
        say(f"  transfer_gain {f(r['transfer_gain_pct'], '.2f')}%: V kept as fitted on the training answers, only its level and strength "
            f"refitted here; at the {f(zero['transfer_gain_pct'], '.1f')}th percentile (random: median "
            f"{f((zero['ref']['transfer_gain_pct'] or {}).get('median'), '.2f')}%)"
            + (f". V saw {trained['units']} training units; {trained['missing']} had no answer" if trained["missing"] else ""))
    if "asis_gain_pct" in r:
        say(f"  asis_gain {f(r['asis_gain_pct'], '.2f')}%: the decider's probabilities taken as they are "
            "(cv_gain minus this is what trusting them costs)")
    if "auc" in r:
        halves = f"half 1 {f(r['auc_h1'])}, half 2 {f(r['auc_h2'])}; " if r["auc_h1"] is not None else ""
        if r["mi_pct"] is None:
            say(f"  AUC {f(r['auc'])} ({halves}taken within the folds: the forecast is fitted out of fold, so MI, the quintiles "
                "and the cross-fit, which compare units of different folds, are left out)")
        else:
            say(f"  AUC {f(r['auc'])} ({halves}{f(zero.get('auc'), '.1f')}th)   "
                f"MI {f(r['mi_pct'], '.2f')}% (p {f(r['mi_p'])}; {f(zero.get('mi_pct'), '.1f')}th)")
            say(f"  positive rate by forecast quintile (low -> high): {', '.join(f(v, '.2f') for v in r['pos_by_quintile_pct'])}")
    d = r.get("decision")
    if d:
        say(f"  as the rule says (skip if forecast >= {d['threshold']}): skip {100 * d['skip_frac']:.1f}%   "
            f"kept positive {f(d['kept_pos_pct'], '.2f')}% (negative {f(d['kept_neg_pct'], '.2f')}%)   "
            f"skipped positive {f(d['skipped_pos_pct'], '.2f')}% (lift {f(d['lift'], '.2f')})")
        say(f"    value per 1,000 units {sign(d['value_per_1000'])}  (take everything {sign(d['all_in_value_per_1000'])}, "
            f"difference {sign(d['gain_vs_all_in_per_1000'])})")
    x = r.get("xfit")
    if x:
        kept = (f"kept {f(x['keep_frac'])}  kept negative {f(x['kept_neg_pct'], '.2f')}%" if x["keep_frac"]
                else "kept 0 (on each half the best was to skip everything)")
        say(f"  cross-fit over the halves (best skip share on one half, applied to the other): value per 1,000 {sign(x['value_per_1000'])}  {kept}")
    for rule in r.get("rule_hits", ()):
        if "hit_frac" not in rule:
            say(f"    rule {rule['rule']}: no expr (the machine cannot answer it)")
        else:
            say(f"    rule {rule['rule']}: hits {100 * rule['hit_frac']:.1f}%  positive when hit {f(rule['pos_pct_hit'], '.2f')}%  "
                f"otherwise {f(rule['pos_pct_rest'], '.2f')}%  lift {f(rule['lift'], '.2f')}")
    if r["calls"]:
        say(f"  decider calls {r['calls']}  cost ${r['cost_usd']:.5f}  errors {r['errors']}")
    say(f"  {SIGN_NOTE}")


def table(rows: Sequence[Mapping[str, Any]], cols: Sequence[tuple[str, Callable[[Mapping[str, Any]], Any], int]], say: Say) -> None:
    """cols: (title, cell of a row, width). The first column is left-aligned, the rest right-aligned."""
    say("  ".join(title.rjust(width) if k else title.ljust(width) for k, (title, _, width) in enumerate(cols)))
    for r in rows:
        cells = []
        for k, (_, cell, width) in enumerate(cols):
            text = "-" if cell(r) is None else str(cell(r))[:width]
            cells.append(text.rjust(width) if k else text.ljust(width))
        say("  ".join(cells))


def _pct(key: str, spec: str = ".2f") -> Callable[[Mapping[str, Any]], str]:
    return lambda r: f(r.get(key), spec)


def _zero(r: Mapping[str, Any]) -> str:
    return f(r["null_pct"]["cv_gain_pct"], ".0f")


def _decision(key: str, fmt: Callable[[float], str]) -> Callable[[Mapping[str, Any]], str | None]:
    return lambda r: fmt(r["decision"][key]) if "decision" in r else None


def comparison(records: Sequence[Mapping[str, Any]], say: Say) -> None:
    """One prompt answered by several deciders, side by side."""
    first = records[0]
    say(f"{first['name']} [{first['prompt_id']}]  split {first['split']}  units {first['n']}  classes {_counts(first['class_counts'])}")
    table(records, [("decider", lambda r: r["decider"], 12), ("model", lambda r: r["model"], 30), ("cv_gain%", _pct("cv_gain_pct"), 8),
                    ("zero pct", _zero, 8), ("asis_gain%", _pct("asis_gain_pct"), 10),
                    ("gap", lambda r: f(r["cv_gain_pct"] - r["asis_gain_pct"], ".2f") if "asis_gain_pct" in r else None, 8),
                    ("auc", _pct("auc", ".3f"), 6), ("refused", lambda r: r["refused"], 7), ("calls", lambda r: r["calls"], 6),
                    ("cost $", lambda r: f(r["cost_usd"], ".5f"), 9)], say)
    say("cv_gain: what V recovers from all the answers. asis_gain: the forecast question's own probabilities. gap: what trusting them costs.")
    say(SIGN_NOTE)


def agree(found: Mapping[str, Any], say: Say) -> None:
    say(f"agreement on the training design: {found['decider']} vs the machine, {found['n']} units "
        f"(a yes/no answer is yes at {found['threshold']} or above, another the option with the most probability)")
    for name, q in found["questions"].items():
        refused = "".join(f"   {who} refused {q[key]}" for who, key in (("model", "model_refused"), ("machine", "machine_refused"))
                          if q[key])
        if "disagreements" in q:
            say(f"  {name.replace('_', ' '):12} agree {f(q['agree_pct'], '.2f')}%{refused}")
            if q["disagreements"]:
                say("      most common disagreements (machine -> model): "
                    + ", ".join(f"{d['machine']} -> {d['model']} {d['n']}" for d in q["disagreements"]))
        else:
            say(f"  {name.replace('_', ' '):12} agree {f(q['agree_pct'], '.2f')}%   both yes {q['both_skip']}  model only {q['model_only']}  "
                f"machine only {q['machine_only']}  neither {q['neither']}   mean answer when the machine says yes "
                f"{f(q['mean_answer_machine_hit'], '.2f')} / no {f(q['mean_answer_machine_miss'], '.2f')}{refused}")
        if q["disagreeing_ids"]:
            say(f"      disagreeing training ids: {', '.join(q['disagreeing_ids'])}")
    if found["calls"]:
        say(f"  decider calls {found['calls']}  cost ${found['cost_usd']:.5f}")


def drift(found: Mapping[str, Any], say: Say) -> None:
    say(f"{found['name']} by {found['by']}, training split only, answers from {found['decider']} ({found['units']} units"
        + (f", {found['missing']} without a stored answer" if found["missing"] else "") + ")")
    if "questions" in found:
        _drift_by_option(found, say)
        return
    rules = len(found["periods"][0]["rule_hit_pct"]) if found["periods"] else 0
    cols = [("period", lambda r: r["period"], 8), ("rows", lambda r: r["rows"], 6), ("pos%", _pct("pos_pct"), 6),
            ("skip%", _pct("skip_pct", ".1f"), 6), ("pos% skipped", _pct("pos_pct_skipped"), 12), ("pos% kept", _pct("pos_pct_kept"), 9),
            ("lift", _pct("lift"), 5), ("value/1k all", lambda r: sign(r["value_per_1000_all"]), 12),
            ("value/1k rule", lambda r: sign(r["value_per_1000_rule"]), 13)]
    cols += [(f"r{j + 1} hit%", lambda r, j=j: f(r["rule_hit_pct"][j], ".1f"), 8) for j in range(rules)]
    table(found["periods"], cols, say)


def _drift_by_option(found: Mapping[str, Any], say: Say) -> None:
    two = len(found["classes"]) == 2

    def outcome(x: Any) -> str | None:
        if x is None:
            return None
        return f(x, ".2f") if two else " ".join(f"{k} {f(v, '.1f')}" for k, v in x.items())

    say("  per option: the share of the period's units, and the outcome among them ("
        + (f"pos% = the share of class {found['classes'][1]!r}" if two else "the share of each class but the first, in %")
        + f"). A yes/no answer is yes at {found['threshold']} or above; another answer is the option with the most probability.")
    width = 6 if two else 26
    for name, q in found["questions"].items():
        say(f"  question {name}")
        cols = [("period", lambda r: r["period"], 8), ("rows", lambda r: r["rows"], 6),
                ("pos%" if two else "classes %", lambda r: outcome(r["outcome"]), width)]
        for j, option in enumerate(q["options"]):
            cols.append((f"{option[:12]} %", lambda r, j=j: f(r["options"][j]["share_pct"], ".1f"), max(8, len(option[:12]) + 2)))
            cols.append(("pos%" if two else "classes %", lambda r, j=j: outcome(r["options"][j]["outcome"]), width))
        table(q["periods"], cols, say)


SURVEY_LEGEND = (
    "per period: logit P(class | F) = a + b x, x the factor's midrank standardized over the units (README.md section 7)",
    "  mu, se   the mean slope across the periods and its standard error (the periods are the replication)",
    "  sigma    the spread of the true slope between periods, sampling noise taken out; below the floor it cannot be told from 0",
    "  share    mu2 / (mu2 + sigma^2), mu2 = mu^2 - se^2: the part of the information that one fixed slope can use",
    "  info%    that part in % of H(Z). nats/yr: how much of it a year of data gathers; a window that gathers 1 nat keeps",
    "           half of it after paying for learning the slope, one that gathers 5 nats keeps 90%",
    "  pq W     the gain in % of H(Z) when the slope is re-learned every month on the last W months (all: every earlier month),",
    "           against re-learning the base rate alone; every window forecasts the same months; * above 2 standard errors",
)


def _exploratory(found: Mapping[str, Any]) -> str:
    return (f"EXPLORATORY: this survey read every split of a closed study ({', '.join(found['splits'])}). Nothing here is evidence for "
            f"this study, and a factor chosen from this table has seen these rows: a new study that uses it needs rows from after "
            f"{found['last']} for its test.")


def _relearned(r: Mapping[str, Any] | None) -> str | None:
    return None if r is None else sign(r["gain_pct"], ".2f") + ("*" if r["gain_pct"] > 2 * r["se_pct"] else " ")


def survey(found: Mapping[str, Any], say: Say) -> None:
    """The header, a legend of the columns, one table per class against the reference, what was not
    surveyed, and the summary. An exploratory survey says so in its first and its last line."""
    if found["exploratory"]:
        say(_exploratory(found))
    source = found["factors_from"]
    what = "the state fields" if source == "fields" else f"prompt {source['prompt']!r} [{source['prompt_id']}], as the machine answers it"
    splits = "training split" if found["splits"] == ["train"] else "every split"
    say(f"SURVEY of {what}: {splits}, {found['first']} .. {found['last']}, every row at weight 1")
    windows = ", ".join("all" if w is None else str(w) for w in found["windows_months"])
    say(f"  {found['units']:,} units in {found['months']} months ({found['units_per_month']:.0f} a month), {found['periods']} periods by "
        f"{found['by']}; re-learning windows {windows} months, gap {found['gap_months']} months")
    for line in SURVEY_LEGEND:
        say("  " + line)
    keys = ["all" if w is None else str(w) for w in found["windows_months"]]
    cols = [("factor", lambda r: r["factor"], 40), ("reading", lambda r: r["reading"], 7), ("periods", lambda r: r["periods"], 7),
            ("mu", lambda r: f(r["mu"], "+.2f") if r["mu"] is not None else None, 6), ("se", _pct("mu_se"), 5),
            ("sigma", _pct("sigma"), 6), ("share", _pct("stable_share"), 5), ("info%", _pct("info_pooled_pct"), 6),
            ("nats/yr", _pct("nats_per_year"), 7)]
    cols += [(f"pq {k}", lambda r, k=k: _relearned(r["relearning"][k]), 7) for k in keys]
    for block in found["by_class"]:
        months = block["predicted_months"]
        say("")
        say(f"  class {block['class']} against {block['against']}: {block['units_in_class']:,} of {block['units']:,} units "
            f"({block['pct']:.2f}%), H(Z) {block['h_bits']:.3f} bits; "
            + (f"re-learning forecasts {months['first']} .. {months['last']} ({months['count']} months)" if months
               else "no month can be forecast (too little history, or a class missing from a window)"))
        table(block["factors"], cols, lambda text: say("  " + text))
        zero = block["no_information_pct"]
        if any(v is not None for v in zero.values()):
            say("  a factor without information is expected at: " + ", ".join(f"pq {k} {sign(v, '.2f')}" for k, v in zero.items()))
        s = block["summary"]
        say(f"  {', '.join(f'{s[r]} {r}' for r in ('stable', 'partly', 'moving', 'none', 'few'))} of {s['factors']} factors; "
            f"chance alone reads about {s['by_chance']} of {s['factors']} as something; "
            f"sigma below {f(s['sigma_floor'], '.2f')} cannot be told from 0 here (median floor)")
        top = s["largest_stable"]
        if top:
            say(f"  largest stable information: {top['factor']} ({top['reading']}), {f(top['info_pooled_pct'], '.2f')}% of H(Z), "
                f"{f(top['nats_per_year'], '.1f')} nats a year")
        best = s["best_relearning"]
        if best:
            say(f"  best re-learning: {best['factor']} on " + ("every earlier month" if best["window"] == "all" else f"the last {best['window']} months")
                + f", {sign(best['gain_pct'], '.2f')}% (se {f(best['se_pct'], '.2f')})" + (" *" if best["starred"] else ""))
    if found["skipped"]:
        say("")
        say("  not surveyed: " + "; ".join(f"{s['factor']} ({s['why']})" for s in found["skipped"]))
    say("  stable means the slope did not move by more than the floor of sigma inside these periods. It says nothing about later ones.")
    if found["exploratory"]:
        say(f"EXPLORATORY (see the first line). Logged in .plab/agents/{found['agent']}/surveys.jsonl; `plab report` counts it.")


def board(rows: Sequence[Mapping[str, Any]], *, split: str, top: int, say: Say) -> None:
    say("training runs (latest per prompt and decider), best cv_gain first. zero = percentile among random answers" if split == "train"
        else "evaluation checks")
    cols = [("name", lambda r: r["name"], 30), ("id", lambda r: r["prompt_id"], 12), ("decider", lambda r: r["decider"], 10),
            ("agent", lambda r: r["agent"], 10), ("cv%", _pct("cv_gain_pct"), 6), ("zero", _zero, 4), ("asis%", _pct("asis_gain_pct"), 8),
            ("auc", _pct("auc", ".3f"), 6), ("h1", _pct("auc_h1", ".3f"), 6), ("h2", _pct("auc_h2", ".3f"), 6), ("mi%", _pct("mi_pct"), 6),
            ("skip%", _decision("skip_frac", lambda x: f(100 * x, ".1f")), 6), ("kept+%", _decision("kept_pos_pct", lambda x: f(x, ".2f")), 6),
            ("val/1k", _decision("value_per_1000", sign), 8), ("vs all", _decision("gain_vs_all_in_per_1000", sign), 8)]
    if split == "train":
        cols.append(("xfit/1k", lambda r: sign(r["xfit"]["value_per_1000"]) if r.get("xfit") else None, 8))
    else:
        cols.append(("transfer%", _pct("transfer_gain_pct"), 9))
    table(rows[:top], cols, say)


def cost(found: Mapping[str, Any], say: Say) -> None:
    spent, budget = found["spent_usd"], found["budget_usd"]
    say(f"spent ${spent:.5f} of ${budget:.2f} (remaining ${max(0.0, budget - spent):.5f})")
    for c in found["by"]:
        say(f"  {c['decider']:10} {c['cmd']:8} {c['agent']:12} calls {c['calls']:6d}  errors {c['errors']:4d}  ${c['usd']:.5f}")


def zero_point(zero: Mapping[str, Any], say: Say) -> None:
    say(f"zero point on {zero['split']} for answers of shape {zero['signature']}: {zero['reps']} random answers of each kind, {zero['rows']} units")
    for kind, metrics in zero["summary"].items():
        for name, s in metrics.items():
            if s:
                say(f"  {kind:6} {name:12} median {s['median']:8.3f}  5th {s['p05']:8.3f}  95th {s['p95']:8.3f}  99th {s['p99']:8.3f}")


def selection(out: Mapping[str, Any], say: Say) -> None:
    rule = out["rule"]
    say(f"selection rule: require {', '.join(q['metric'] + ' ' + q['op'] + ' ' + str(q['value']) for q in rule['require'])}; "
        f"rank by {', '.join(k['metric'] + ' ' + k['order'] for k in rule['rank_by'])}")
    for c in out["candidates"]:
        train, ev = c["train"], c["eval"]
        d = ev.get("decision")
        value = (f"  eval value/1k {sign(d['value_per_1000'])} (take-everything {sign(d['all_in_value_per_1000'])}, "
                 f"difference {sign(d['gain_vs_all_in_per_1000'])})  skip {100 * d['skip_frac']:.1f}%" if d else "")
        say(f"  {'PASS' if c['passed'] else 'fail'}  {c['name'][:40]:40} [{c['prompt_id']}] {c['decider']:10} "
            f"train cv {f(train['cv_gain_pct'], '.2f')}%  eval cv {f(ev['cv_gain_pct'], '.2f')}% "
            f"(zero {f(ev['null_pct']['cv_gain_pct'], '.0f')}th)  eval transfer {f(ev['transfer_gain_pct'], '.2f')}%{value}")
        for failed in c["failed"]:
            say(f"        fails: {failed}")
    chosen = out["selected"]
    say(f"selected: {chosen['name']} [{chosen['prompt_id']}] with decider {chosen['decider']}" if chosen else f"selected: none. {out['reason']}")


def test(res: Mapping[str, Any], say: Say) -> None:
    say(f"TEST (one run): {res['name']} [{res['prompt_id']}] decider {res['decider']}"
        + (f", skip if forecast >= {res['threshold']}" if "threshold" in res else ""))
    info = res["info"]
    zero = info["null_pct"]
    say(f"  units {info['n']}  classes {_counts(info['class_counts'])}" + (f"  refused {info['refused']}" if info["refused"] else ""))
    say(f"  cv_gain {f(info['cv_gain_pct'], '.2f')}% at the {f(zero['cv_gain_pct'], '.1f')}th percentile of {zero['kind']} random answers"
        + (f"   transfer_gain {f(info['transfer_gain_pct'], '.2f')}% ({f(zero['transfer_gain_pct'], '.1f')}th)"
           if "transfer_gain_pct" in info else "")
        + (f"   asis_gain {f(info['asis_gain_pct'], '.2f')}%" if "asis_gain_pct" in info else "")
        + (f"   AUC {f(info['auc'])} ({f(zero.get('auc'), '.1f')}th)" if "auc" in info else ""))
    for label, key in (("selected prompt", "selected"), ("take everything", "take_everything")):
        r = res.get(key)
        if r:
            say(f"  {label:16} taken {r['taken']}/{r['rows']}  "
                f"positive {f(r['taken_pos_pct'], '.2f')}% {r['taken_pos_ci95']}  negative {f(r['taken_neg_pct'], '.2f')}% {r['taken_neg_ci95']}  "
                f"value/unit {f(r['value_per_taken'])}  total {sign(r['value_total'])}"
                + (f"  AUC {r['auc']} (p {r['auc_p_one_sided']})  MI {r['mi_pct']}% (p {r['mi_p']})" if "auc" in r else ""))


def report(rep: Mapping[str, Any], say: Say) -> None:
    say(f"REPORT: {rep['task']} ({rep['phase']})")
    say(f"  pre-registration: {rep['prereg']}")
    say(f"  the evaluation set was looked at {rep['eval_looks_total']} times in all:")
    for agent, n in sorted(rep["eval_looks_by_agent"].items()):
        say(f"    {agent:16} {n}" + ("  (the selection step, by the rule)" if agent == "select" else f" / {rep['eval_checks_cap_per_agent']}"))
    say(f"  training-set runs: {rep['train_runs']}")
    say(f"  exploratory surveys of every split (after the test): {rep['exploratory_surveys']}")
    say(f"  nominations: {len(rep['nominations'])}")
    for r in rep["nominations"]:
        say(f"    {r['agent']}: {r['name']} [{r['prompt_id']}] {r['decider']}")
    chosen = rep["selected"]
    if not rep["selection_done"]:
        say("  selection: not run")
    elif chosen:
        say(f"  selection: {chosen['name']} [{chosen['prompt_id']}] decider {chosen['decider']} (nominated by {chosen['nominated_by']})")
    else:
        say(f"  selection: none ({rep['selection_reason']})")
    if rep["test"]:
        test(rep["test"], lambda text: say("  " + text))
    else:
        say("  test: not run")
    for decider, c in rep["cost"].items():
        say(f"  cost {decider}: {c['calls']} calls, ${c['usd']:.5f}")
    say(f"  budget: ${rep['budget_usd']}")
    for name, summary in rep["eval_zero_point"].items():
        say(f"  zero point on the evaluation set ({name}, cv_gain_pct of random answers): "
            + "; ".join(f"{kind} median {s['cv_gain_pct']['median']}%, 95th {s['cv_gain_pct']['p95']}%" for kind, s in summary.items() if s["cv_gain_pct"]))
