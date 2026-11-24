# prompt-lab

Start here: the theory behind every yardstick opens [README.md](README.md) (in Japanese).
It explains the idea every yardstick in this harness rests on: from the probabilities a decision
model returns, measure how many more bits about the outcome a prompt lets you recover.

prompt-lab measures, for any task with a known outcome, how much a prompt lets a **decision model**
tell about that outcome, **without fooling yourself**. You give it timestamped cases with an outcome
(two classes or more, and optionally a value such as +2 for a win and −50 for a loss). Searchers
(people or AI agents) write prompts: which fields of a case the model reads, and which typed questions
it answers about them. The harness measures each prompt on a training period, lets each searcher look
at an evaluation period only a limited, logged number of times, selects by a rule written down in
advance, and runs the selected prompt on a test period exactly once.

The first use, and the case study behind the tool, is *veto rules*: "skip this case if any of these
apply". The same yardsticks measure a multi-class outcome or a prompt with several questions.

It was built for [Jev](https://vercel.com/docs/ai-gateway/modalities/decision) (`typesafe-ai/jev` on
the Vercel AI Gateway), a decision model that takes a state and typed questions and returns
probabilities. A rule or a question whose answer can be written as a Python-like expression is also
answered by the free `machine` decider, so most of the search costs nothing.

Requirements: Python 3.10+, numpy. No other dependency.

[日本語の README](README.md) · [The protocol and its reasons](docs/protocol.md)

## Why

Searching prompts by trial and error on labelled data overfits quickly, and the overfitting is hard
to see. In the case study behind this tool (FX alerts), every one of these happened:

- **Losses cluster in time.** Cross-validation that splits row by row leaks a losing episode into both
  sides; splitting by month made a tree model's AUC drop from 0.63 to 0.57.
- **"Positive" is not zero.** Random answers have a cross-validated gain around −0.7% of H(Z) on a
  1,270-row evaluation set; a result is only evidence when it sits high in that random distribution.
- **Looking is fitting.** Every check on the evaluation set is a chance to pick what happened to work there.
- **AUC is not money.** With +2 wins and −50 losses you break even at a 96.2% win rate; a better AUC
  can still lose. The cross-validated gain is even sign-blind: a rule whose effect *reversed* can score
  high on it.
- **Regimes change.** Trend and volatility rules that worked in both halves of the training period
  vanished on the evaluation period; only narrow structural rules (fixed time windows) survived.

The harness turns these into mechanics: time splits, month-grouped CV, a measured zero point, a cap on
evaluation looks with a log, a frozen pre-registration, rule-based selection and a one-shot test.
[docs/protocol.md](docs/protocol.md) explains each one.

## Quick start (no key needed)

```sh
git clone <this repository> prompt-lab && cd prompt-lab
uv venv && uv pip install -e .          # or: python -m venv .venv && .venv/bin/pip install -e .
python examples/synthetic/walkthrough.py --reset    # use the venv's python
```

The walkthrough makes 12,000 synthetic alerts (2018–2025, about 4.6% fail, a win is +2 and a failure
−50) with three planted effects: (a) alerts between 13:00 and 15:59 UTC fail 1.6 times as often, in
every year; (b) a strong move toward the alert direction makes failure 2.4 times as likely before
2022 and *less* likely after; (c) volatility grows 2.5 times in 2022. Splits: training 2018–2021,
evaluation 2022–2023, test 2024–2025. It then runs every command below, with the `machine` decider and
the `fake` decider. What you will see (seed 7):

| prompt (machine decider) | training: cv_gain (percentile among random answers), AUC halves | evaluation: AUC, value per 1,000 vs take-everything |
|---|---|---|
| (b) skip when `move_24h_toward > 12` | 0.69% (100th), 0.528 / 0.571, lift > 1 in every year | 0.464 (3rd percentile), **−78**; skips 12% → 29% because volatility grew |
| (b) relative to volatility | | cv_gain 0.54% (100th) but AUC 0.460, **−140**: information with the wrong sign |
| (a) skip 13:00–15:59 UTC | 0.52% (100th), 0.562 / 0.553 | 0.565, cv_gain 0.56% (100th), **+356** |
| (a) or (b) | 1.12% (100th), the best in training | cv_gain −0.19% (74th), fails selection |

The selection rule picks (a); the single test run gives 3.84% failures among the 2,242 alerts taken
(Wilson 95%: 3.1–4.7%) against 4.73% for taking all 3,042, and +12 in total instead of −1,404. A second
`plab test --final` is refused. Across data seeds 1–10, (a) beats take-everything on the evaluation set
in 10 of 10 and has a positive evaluation cv_gain in 7 of 10; (b) loses on the evaluation set in 10 of 10.

Each run also prints `asis_gain`, the gain of the decider's own probabilities. The machine answers
exactly 0 or 1, so its as-is gain is far below zero (−1,991.51% for (a) in training): a sure forecast
that is wrong costs about 20 bits. The recalibrated `cv_gain` does not pay that cost.

## Using Jev

1. Set the key in the environment: `AI_GATEWAY_API_KEY=...`, or `PLAB_KEY_FILE=/path/to/a/file/with/the/key`.
   The key is never printed, logged or written to results (error texts are scrubbed). Do not copy the key
   file into a task folder or a repository.
2. In `task.json`, name the decider and set a budget. The default `deciders` already holds
   `"jev": {"kind": "gateway", "model": "typesafe-ai/jev", "price_per_mtok": 0.042}`, and `"decider": "jev"`
   is the default. Set `budget.usd`.
3. Before calling, the harness estimates the cost and refuses if it exceeds the remaining budget
   (`--yes` overrides). Real costs (from the gateway's response) go to the ledger,
   `.plab/agents/<agent>/ledger.jsonl`; `plab cost` sums them.
4. Each case is one call to `POST /v1/decisions`: the fields the prompt gives, as one JSON part, and the
   questions. Answers are stored under `.plab/answers/`, keyed by the request, so cases that send the same
   request share one call. An interrupted run continues
   where it stopped, a request is never paid for twice, and editing an `expr` or a forecast reuses the
   answers. `--cache-only` refuses to call at all.
5. Calls: `gateway.workers` (6) in parallel. 429, 5xx and 529 are retried `gateway.retries` (4) times,
   waiting as `Retry-After` says or 1, 2, 4 ... seconds. 401, 402 (no credit) and 403 stop the run, and so
   does a reply that answers other questions; what was paid for so far is stored.

A typical full evaluation (1,000–1,300 calls of ~600 tokens) costs about $0.03–0.04.

To compare decision models on one prompt, add them to `deciders` and name several:
`plab run P.json --agent A --decider jev --decider other`.

## The study, step by step

```
plab init mytask            # task.json, prereg.md (template), prompts/
#   put rows.jsonl in mytask/, edit task.json
plab -C mytask split        # .plab/data/{train,eval,test}.jsonl by time
plab -C mytask fields --auc # what is in the state (training split only)
plab -C mytask survey       # does each factor's relation to the outcome hold across periods? (training split, free)
#   searchers: write prompts, then
plab lint P.json                            # wordings that lower model/machine agreement
plab run P.json --decider machine --agent A # free: the exprs answer, on the training split
plab run P.json --agent A                   # the task's decider on the training split
plab agree P.json --agent A [--per-rule]    # does the model apply the rules like the machine?
plab drift P.json --by year                 # does the rule hold in every period of training?
#   the person running the study, before anyone looks at the evaluation split:
plab prereg freeze
#   searchers:
plab check P.json --agent A [--decider NAME]     # ONE look at the evaluation split; capped per agent
plab nominate P.json --agent A [--decider NAME]  # put forward at most N candidates
#   the person running the study:
plab select                 # applies the frozen selection rule; writes .plab/selected.json or "none"
plab test --final           # the selected candidate on the test split, once
plab report                 # looks at the evaluation split, selection, test table, cost
```

What is measured, nominated and selected is a candidate: a prompt and a decider. `plab brief --agent A
--direction "..."` prints the instructions to hand to a searching agent: what it may read, what it must
not open, which commands it may run, its caps, and how to report back. In Claude Code, the skill
[plab-search](.claude/skills/plab-search/SKILL.md) (in Japanese) takes that brief and keeps writing,
measuring and revising prompts until it nominates. Before any search, the skill
[plab-survey](.claude/skills/plab-survey/SKILL.md) runs `plab survey` and decides whether a search is
worth starting.

Commands take `-C DIR` (the task folder; default: the current folder). Refusals exit with code 2.

| command | reads | notes |
|---|---|---|
| `init DIR` | – | writes templates; `prereg.md` carries a mark that `freeze` refuses until you fill it in |
| `migrate [--write] [--unfreeze]` | – | converts a prompt-lab 0.1 task folder (see below) |
| `split [--force]` | rows.jsonl | refused after freeze |
| `fields [--auc]` | train | fill rate, quantiles, top values; `--auc` (two classes): single-field AUC on the weighted training sample |
| `lint PROMPT` | train | warnings (lesson 3), an expr compared with None (it never holds), what the machine cannot answer, fields named but not given |
| `run PROMPT [--decider NAME ...]` | train | logged to `runs.jsonl`; several `--decider` print the deciders side by side |
| `agree PROMPT [--decider NAME] [--per-rule] [--limit N]` | train | every question the machine can answer (rules with an `expr` on each, or an `expr`): a yes/no answer at the decision threshold, a choice or score by the option it gives the most probability, against the machine, leaving out units either one refused; disagreeing ids |
| `drift PROMPT [--decider NAME] [--by quarter\|half\|year]` | train | by period: with `decision` and a forecast, rule hits, positive rates and value; otherwise each question's answers by option and the outcome among them |
| `survey [PROMPT] [--by quarter\|half\|year] [--gap MONTHS] [--splits train\|all]` | train (`all`: every split) | the state fields, or a prompt's rules and questions as the machine answers them, as factors: each one's slope by period, its mean and spread, and the gain of re-learning it (see the yardsticks). Writes nothing on the training split; `--splits all` runs only on a closed study, is marked EXPLORATORY and is logged to `surveys.jsonl` |
| `null PROMPT [--split train\|eval] [--force]` | train, eval | the zero point for the prompt's answer shape (stored per split and shape); `eval` needs freeze |
| `prereg freeze [--force]\|status` | – | hashes prereg.md, the locked settings of task.json and the split files |
| `check PROMPT --agent A [--decider NAME]` | eval | freeze required; cap `budget.eval_checks_per_agent`; every check logged |
| `nominate PROMPT --agent A [--decider NAME]` | – | cap `budget.nominations_per_agent`; stores the prompt itself |
| `select` | train, eval | re-measures each nomination; its looks are logged as agent `select` |
| `test --final [--resume]` | test | refused without a selection, and the second time |
| `board [--split train\|eval] [--agent A] [--all] [--top N]` | logs | the evaluation board shows one agent's own checks unless `--all` |
| `cost`, `report [--json]`, `brief --agent A [--direction TEXT]` | logs | |

Flags on measuring commands: `--agent NAME`, `--decider NAME`, `--cache-only`, `--yes`, `--json`.

### Convert a prompt-lab 0.1 folder

`plab -C OLD migrate` prints what it would change; add `--write` to do it. It rewrites `task.json` into
format 2 from the 0.1 settings and 0.1's own defaults (the old file is kept as `task.v1.json`), moves the
answers 0.1 paid for into the store (marked `via: "evaluate"`), and makes the 0.1 ledger the ledger of the
agent `migrate`, so it still counts against the budget. Prompt files and split files stay as they are.
A score question's answers are not moved, because 0.1 kept only score/(levels − 1). The old cache, logs
and zero points are set aside under `.plab/archive-v0.1/`. A study frozen under 0.1 is refused unless
you pass `--unfreeze`, which also sets aside its lock, checks, nominations and selection; a study that
started its test is never migrated. When one table holds answers from both endpoints, the result prints the
count from each.

## Files

### rows.jsonl (one case per line)

```json
{"id": "a00017", "t": "2018-01-03T14:05:00Z", "state": {"time_utc": "14:05", "vol_24h": 18.2, "...": "..."}, "label": 1, "value": -50.0}
```

The outcome is one of `outcome.classes` (`true` and `false` read as 1 and 0, a whole number as its
integer, then matched as text). `value` is what taking the case is worth; leave it out, and set
`rows.value` to `null`, when cases have none. `t` is ISO 8601 or epoch seconds (naive times are read as
UTC). Keep dates, years and absolute prices out of the state if the model might remember what happened.

### task.json

`"format": 2` is required. Every other key has a default; objects merge into the default, lists replace
it, and `selection` replaces it whole. Keys starting with `_` are ignored (use them for notes).

| key | meaning (default) |
|---|---|
| `name` | the study's name (`task`) |
| `rows` | `file`, and the names of `id`, `time`, `outcome`, `value` and `state` in a row (`rows.jsonl`, `id`, `t`, `label`, `null`, `state`) |
| `outcome` | `classes`: two or more, in index order (`["0", "1"]`); `meaning`: free text per class, shown to searchers. Class 0 is the reference class, the common case. With two classes, class 1 is the positive class |
| `splits` | `{"train": [start, end), "eval": [...], "test": [...] or absent}`, in time order |
| `cv_groups` | `month`, `week`, `day`, `row` or a state field: rows of one group stay in one CV fold (`month`) |
| `half_boundary` | training rows before it are half 1, the rest half 2 (for half AUCs and the cross-fit) |
| `train_sample` | `{"reference_per_other": 6, "seed": 12345}`: every row outside class 0, plus N class-0 rows for each, weighted back (`null`: every row) |
| `decision` | the veto use, `{"skip_if_prob_at_least": 0.5}`: needs two classes and `rows.value` (`null`) |
| `state_fields` | `"all"` or a list: the fields a prompt may give to the decider |
| `field_groups` | named lists of fields, usable as a prompt's `fields` |
| `rule_frame` | `head` and `tail`: the text around the rules of a rules question (generic English; rewrite it for your domain) |
| `selection` | `require`: list of `{metric, op, value}`; `rank_by`: list of `{metric, order}` (see below) |
| `budget` | `usd` (0), `eval_checks_per_agent` (10), `nominations_per_agent` (2) |
| `deciders` | name → `{"kind": ...}`: `machine` (no settings), `fake` (`cost_per_call`), `gateway` (`model`, `price_per_mtok`). Defaults: `machine`, `fake`, `jev` |
| `decider` | the decider a command uses when `--decider` is not given (`jev`) |
| `gateway` | `workers` (6), `retries` (4), `timeout_s` (120) |
| `metrics` | `cv_folds` (10), `cv_reps` (3), `mi_perms` (500), `null_reps` (200), `null_seed` (777) |

`prereg freeze` locks every setting except `name`, `budget.usd`, `gateway` and the prices of the deciders.

A selection metric is `train.` or `eval.` and then a record field: `cv_gain_pct`, `asis_gain_pct`,
`null_pct.cv_gain_pct`, `n`, `missing`, `refused`, `class_counts.<class>`; on evaluation also
`transfer_gain_pct` and `null_pct.transfer_gain_pct`; with two classes also `auc`,
`mi_pct`, `mi_p`, `null_pct.auc`, `null_pct.mi_pct`, and `auc_h1`, `auc_h2` (training, with
`half_boundary`); with `decision`, `decision.value_per_1000` and the other numbers of the decision table;
with two classes, values and `half_boundary`, `xfit.value_per_1000`, `xfit.keep_frac`,
`xfit.kept_neg_pct` (training). Loading refuses a rule that reads a number the task cannot produce. The
default rule requires a positive cv_gain on both splits and a positive evaluation value, so a task
without `decision` must state its own.

### Prompts

```json
{"name": "afternoon window",
 "fields": ["event", "side", "time_utc"],
 "questions": [{"name": "skip", "type": "predicate", "rules": [
   {"text": "(a) Skip when time_utc is between 13:00 and 15:59 inclusive.", "expr": "'13:00' <= time_utc <= '15:59'"},
   "(b) A rule in words only, without expr."]}],
 "forecast": {"question": "skip", "classes": {"yes": "1", "no": "0"}}}
```

- `fields`: `"all"`, a name from `field_groups`, or a list of state fields. The decider reads these
  fields only.
- `questions`: 1 to 6, each with a `name` and a `type`:
  - `predicate`: yes or no. Either `instructions` and an optional `expr`, or `rules`: 1 to 12 rules,
    each a text or `{"text", "expr"}`. A rules question is asked as `rule_frame.head` + the rule texts
    + `rule_frame.tail`, and the machine says yes when any rule applies.
  - `choice`: `instructions` and `choices`, 2 to 255 options, each a text or `{"value", "description"}`.
    An `expr` gives the option the machine chooses.
  - `score`: `instructions` and `levels`, 2 to 10, each a text or `{"label", "description"}`. An `expr`
    gives the level index.
- `forecast`: what the answers say about the outcome.
  - `{"question": NAME, "classes": {option: class}}`: every option of one question forecasts a class.
    The decider's own probabilities are then a forecast, which gives the as-is gain and, with two
    classes, AUC, binned MI and the decision table.
  - `"fitted"`: the forecast is V fitted on the training answers. On the training split it is out of fold, so its AUC is taken
    within each fold and averaged, and binned MI, the quintiles and the cross-fit, which compare units of different folds, are
    left out. Pooled over folds they read a weak answer wrong: its AUC fell below 0.5 and its MI rose.
    There is no as-is gain.
  - absent: the prompt is measured by cv_gain alone.

A task with three classes might ask a choice and a predicate:

```json
{"name": "route", "fields": ["topic", "x"],
 "questions": [{"name": "queue", "type": "choice", "instructions": "Which queue should take this ticket?",
                "choices": ["other", {"value": "billing", "description": "refunds and payments"}, "shipping"], "expr": "topic"},
               {"name": "big", "type": "predicate", "instructions": "Is x above 1?", "expr": "x > 1"}],
 "forecast": {"question": "queue", "classes": {"other": "other", "billing": "billing", "shipping": "shipping"}}}
```

The two prompt forms of prompt-lab 0.1 still load for a two-class task: `veto_rules` (one rules
predicate whose yes forecasts class 1), and `questions` as an object beside `score`.

**`expr`** is a small, safe language (parsed with `ast`, never `eval`): field names, numbers, strings,
`True/False/None`, tuples/lists, comparisons (`== != < <= > >= in, not in`, chained), `and/or/not`,
`+ - * /`, `abs/min/max/len`, and subscripts/slices. An `expr` may only read fields that the prompt gives
to the decider. **If a field it reads is null (or a list element it uses is null), or the computation fails
(type mismatch, division by zero), the expr has no value: a rule does not apply, a predicate says no, and a
choice or score counts as refused.**

The prompt id is a hash of the fields, the questions as asked, their exprs and the forecast, not of the
name. The answer store is keyed by what the decider reads: a gateway model reads the request only, so
editing an expr or the forecast reuses its answers.

### .plab/ (written by the harness, append-only where it is a log)

| file | content |
|---|---|
| `data/{train,eval,test}.jsonl` | the split rows, in time order |
| `answers/<shelf>/meta.json`, `answers/<shelf>/<agent>.jsonl` | what one shelf of stored answers was asked (model and questions), and the answers each agent fetched |
| `nulls/<split>.<signature>.json` | the zero point of one split and answer shape |
| `agents/<agent>/{runs,checks,nominations,agree,ledger,surveys}.jsonl` | the logs, one file per agent: training runs, looks at the evaluation split (agents and `select`), nominations (with the prompt itself), agreement checks, decider calls with their cost, surveys of every split of a closed study |
| `prereg.lock`, `prereg.frozen.md` | the frozen hashes and a copy of prereg.md |
| `selected.json` | the selection |
| `test.lock`, `test_result.json` | the one test run |

## The yardsticks

Z is the outcome with K ≥ 2 classes; class 0 is the reference. Each unit has a weight w (1 everywhere
except in a down-sampled training split). The theory at the top of [README.md](README.md) gives the reasons.

**The answers, encoded.** An answer is a probability over a question's options. A predicate gives
logit(P(yes)), a choice gives log(p_j / p_0) for each option after the first, and a score gives one
column, logit(mean level / (levels − 1)). Probabilities are clipped to [1e-4, 1 − 1e-4] first. A choice
answered without probabilities is a point mass on the chosen option. A refused question counts as the
uniform answer: its unit stays in, and the record counts it in `refused`.

Information, for any K and any number of questions:

- **cv_gain_pct**, the recalibrated gain: V is a multinomial logistic regression (softmax, class 0 as
  reference) on the encoded answers of every question, fitted out of fold (10 folds grouped by
  `cv_groups`, 3 repeats, seeds 100 + r, a 1e-3 ridge on the slopes). The result is
  100 · (H(Z) − cross-entropy) / H(Z). It estimates the V-information of the answers about the outcome.
  With one predicate and two classes it is exactly the number 0.1.0 gave. **Sign-blind.**
- **transfer_gain_pct**, the gain in the training direction (evaluation and test): V is fitted on all
  training answers, and its slopes give each case a score E per class. On the split measured, E stays
  fixed and only the intercepts and one strength c ≥ 0 are fitted, out of fold with the folds of
  cv_gain. An answer whose relation to the outcome turned around falls to the zero point; one whose
  relation only weakened keeps a positive gain. It fits one number more than the constant forecast,
  whatever the number of columns. Measuring evaluation or test also reads the training answers (asking
  the decider for those not stored); the record counts those without an answer (`trained_on`). **Sign-aware.** See
  [ADR-0011](docs/adr/0011-transfer-gain-keeps-the-training-direction.md).
- **asis_gain_pct**, the as-is gain: the same share for the decider's own probabilities, through the
  prompt's `forecast`. Nothing is fitted, so it can be below zero on any number of units;
  cv_gain − asis_gain is what trusting the raw probabilities costs. **Sign-aware.**

Use, for two classes and a forecast (s is the forecast probability of class 1):

- **AUC** (weighted, ties count half), and on each training half. **Sign-aware.**
- **mi_pct**, binned MI: s in 5 bins of equal weight, the mutual information with Z minus the mean of 500
  shuffles, in % of H(Z), with its permutation p. It fits no model. **Sign-blind.**
- **Positive rate by forecast quintile.**
- **Decision** (with `decision` and values): skip when s ≥ the threshold. The skip share, the positive and
  negative rates among the cases taken, the positive rate among those skipped and the lift, the value per
  1,000 cases, and the same for taking everything. **Sign-aware.**
- **Cross-fit** (training, with values and `half_boundary`): the best skip share (5% steps) chosen on one
  half and applied to the other, both ways. **Sign-aware.**
- **Test**: cases taken, positive and negative rates with Wilson 95% intervals, value per case and in
  total, AUC with a one-sided permutation p, and binned MI; next to taking everything.

**Zero point.** The same yardsticks on random answers of the prompt's shape (the kind and number of
options of every question and, with two classes, which options forecast class 1), keeping the split's labels, weights and
groups. There are two kinds, 200 of each: *soft* answers spread their probability at random, and
*sharp* answers put all of it on one option. A result is placed among the sharp kind when at least 90%
of its weight has at most 0.15 of its probability off one option on every question, and among the soft
kind otherwise. Every record prints its percentile. On evaluation and test, the direction of
`transfer_gain_pct` comes from V fitted on other random answers.

### Information that holds across time (`plab survey`)

Section 7 of [README.md](README.md) is the source; this is its English counterpart. Here t is
time and F a factor: one number per case, such as a state field, whether a text field takes a value,
or a rule or question of a prompt as the machine (`expr`) answers it. (The T of the theory above is
what the decider reads, a different thing.)

**The split.** I(Z;F|t) = I(Z;F) + [I(Z;F|t) − I(Z;F)], and the bracket equals I(Z;t|F) − I(Z;t). The
first term is what one relation over all periods can use; the bracket rides on how the relation
changes with time (it is −I(Z;F) when F and Z are independent given t). A search that writes fixed
rules reaches the first term only.

**A stale slope.** A forecast q fitted at another time scores gain(q) = I_t − E KL(p_t ‖ q), I_t =
I(Z;F|t), against the period's own base rate. With F standardized, a class share π and a weak effect,
I_t ≈ κ b_t² with κ = π(1 − π)/2 nats. A slope fixed at b (intercept refitted) loses about κ(b − b_t)²,
so it keeps κ(2 b b_t − b²). With μ and σ² the mean and the between-period variance of the true slope,
the best fixed slope is b = μ and keeps κμ² on average, of the κ(μ² + σ²) a slope known in every period
would use: the stable share μ²/(μ² + σ²). With several columns, the angle θ between the learned and the
current coefficients keeps I_t cos²θ; `transfer_gain_pct` refits a strength c ≥ 0, so for one factor
only a reversed sign costs. At π = 5%, κb² is within 0.7% of the exact I(Z;F) (Gauss-Hermite) at
b = 0.3 and within 2.1% at b = 0.5; it overstates by 13% at b = 1 (32% at π = 40%), so read strong
factors by the re-learning columns.

**The estimate.** The factor's midranks (ties share their mean rank; a case without a value sits at the
mean rank), standardized: rows are in time order, so ranking ties by position would plant a time trend.
A text field gives one 0/1 factor per value held by 2% to 98% of the cases, when it has at most 12
values; lists and rules without an `expr` are listed as not surveyed. In every period (`--by`, half a
year by default) a one-factor logistic slope b_t and its Fisher standard error s_t, on every row of the
split at weight 1 (the factors are free, so nothing is down-sampled); a period needs 5 cases of each
class. DerSimonian-Laird across the periods: Q, σ̂² = max(0, (Q − (k − 1)) / (Σw − Σw²/Σw)) with
w = 1/s², μ weighted by 1/(s² + σ̂²) and its SE. μ² is biased up by SE(μ)², so the survey uses
μ₂ = max(0, μ² − SE(μ)²): info% = 100 κ μ₂ / H(Z) (in nats), share = μ₂/(μ₂ + σ̂²), nats/yr = a year of
cases times κμ₂. σ̂ is biased up near 0; the floor of σ is where Q is expected to reach the spread test,
and a smaller σ cannot be told from none with these periods. With K > 2 classes, each class is surveyed
against the reference class on the cases of the two.

**What can be learned.** Fitting one slope on N cases costs about 1/(2N) nats per forecast case, so a
window keeps 1 − 1/(2NI) of I: half when it gathers N·I = 1 nat, 90% at 5 nats. A random-walk slope
that drifts its own size in τ periods, with n cases a period, can be tracked to about 1 − 1/√(2nτI).
The survey measures re-learning directly (`pq`, prequential as in Dawid 1984): every month is forecast
by a + bF fitted on the 6, 12 or 24 months before it, or on all of them, against the base rate
re-learned on the same window, in % of H(Z) of the forecast cases. `--gap` is the months before an
outcome is known. Every window forecasts the same months (those with the longest fixed window plus the
gap before them, both classes in every window), so the columns compare; a fixed window is kept only
when it is at most half the span. The standard error comes from the spread of the monthly gains (losses
cluster in time) and `*` marks a cell above 2 of them. A factor without information is expected at
−100 Σ n_m/(2N_m) / (n H(Z)); in the planted checks the 24-month and all-month windows land there and
the short windows a little below (0.02 to 0.09 points on 6 months).

**Reading.** The mean counts when |μ| ≥ 2 SE(μ); the spread counts when Q as a standard normal
(Wilson-Hilferty) is at least 2. Mean and no spread: `stable`; both: `partly`; spread only: `moving`;
neither: `none`; fewer than 4 periods: `few`. A factor without information reads as something about
6.7% of the time, and the summary prints that count next to the readings; nothing is corrected for
multiplicity, and a rule found by searching the same rows has an inflated μ. `stable` says the slope did
not move by more than the floor inside the surveyed periods, and nothing about later ones. The survey
gives no verdict on a study; the [plab-survey](.claude/skills/plab-survey/SKILL.md) skill decides.

**Checked.** Planted slopes, 95 cases a month at a 5% rate, 40 seeds over 20 half-years: a slope fixed
at 0.3 reads `stable` 40 times; mean 0.3 with spread 0.3, `partly` 37; spread 0.4 around 0, `moving`
34; no information, `none` 37 and no starred cell of 160. Over 8 half-years (a four-year training
split) the floor of σ is 0.23, so a spread of 0.3 reads `stable` 15 times out of 40. In the synthetic
walkthrough, rule (b), which flips in 2022, reads `stable` on the training split and `moving` on every
split after the test (σ̂ 0.20, share 0.11). On the FX case study's rows (2022-01 to 2026-10, 5,170
cases, 242 losses), 6 of 47 factors read as something against 3.2 by chance, the largest fixed-slope
information is 0.24% of H(Z), or 0.5 nats a year, and no re-learning cell is starred.

## What the harness enforces, and what it cannot

Enforced: the time split; that checks, selection and the test need a frozen, unchanged pre-registration;
the per-agent caps; the log of every look; that selection follows the rule; that the test runs once and
only for a selection; the budget; that an `expr` reads only what the decider sees.

Not enforceable by software: that nobody opens `rows.jsonl` or `.plab/data/eval.jsonl` directly. The brief
tells searching agents not to, and the logs make looks through the tool countable. Treat the eval and test
files like a sealed envelope.

## Tests

`python -m unittest` (no network, no key; about 35 seconds).

## License

MIT. See [LICENSE](LICENSE).
