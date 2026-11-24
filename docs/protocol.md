# The protocol, and why

This harness encodes seven lessons from a case study on FX alerts: a judge model was asked, alert by
alert, whether any of a set of veto rules applied, and several AI agents searched for veto rules that
would improve the profit of the alerts that were taken. The search mostly failed, and *how* it failed is
what the harness is built around. Each lesson below says what happened, the general rule, and what the
harness does about it.

## 1. Split by time, and keep time together inside cross-validation

**What happened.** Losses came in episodes: several alerts in the same few days lost together. When
cross-validation assigned alerts to folds one by one, an episode landed on both sides of a split and the
model looked better than it was. Assigning whole calendar months to folds lowered a gradient-boosted
tree model's AUC from 0.63 to 0.57 on the same data.

**Rule.** Separate training, evaluation and test periods by time, in that order. Inside a period, put
rows that share an episode (a month, a week, a day, or a field you name) in the same fold.

**Harness.** `split` assigns rows to `[start, end)` periods from `task.json` and refuses overlaps.
`cv_groups` (default `month`) groups the folds of every cross-validated number. The training set is
split into two halves in time (`half_boundary`) so that a result can be checked on each half.

## 2. Measure where zero is

**What happened.** The main yardstick, the cross-validated information gain (`cv_gain_pct`, the drop in
cross-entropy of an out-of-fold logistic regression on the answer, in % of H(Y)), is negative for an
uninformative answer, and by how much depends on the data. On the evaluation set of the case study
(1,270 alerts, 54 losses, 16 months), random answers gave a median of −0.72% and a 95th percentile of
about 0. A result of "−0.1%" is therefore not "no information"; it is near the top of what chance gives.

**Rule.** Report every number next to its place in the distribution that random answers produce on the
same rows, labels, weights and groups.

**Harness.** `null` (run automatically when needed) scores 200 soft and 200 sharp random answers
per split and answer shape and saves the distribution. Every run, check and selection prints the percentile among random
answers of the matching kind. Selection rules can require it (`eval.null_pct.cv_gain_pct`).

## 3. Search with the machine; use the model where words are needed

**What happened.** When a veto rule was written with numbers ("skip when the band width is below 12"),
the judge model applied it the same way as the identical condition written in Python on 99–100% of
alerts. It added no information of its own. Agreement fell to 59–95% for three kinds of wording:
multiplying by a number below 1 ("more than 0.4 times the band width"), taking the maximum over a list of
12 values, and prompts with four or more rules that contain "and". In those cases the model leaned toward
skipping near the boundary. Rules written only in words lost information compared with the same rule in numbers.

**Rule.** Write rules that can be written with numbers as numbers, and evaluate them by machine while
searching; it is free and exact. Spend model calls on rules that need words and on checking that the
model applies the numeric rules as written.

**Harness.** Each rule may carry an `expr` in a small safe language. `run --decider machine` scores a
prompt with the machine at no cost; `agree` (optionally rule by rule) measures how often the model's decision matches
the machine's; `lint` warns about the three wordings above.

## 4. Every look at the evaluation set is a fit

**What happened.** Agents were allowed to check their prompts on the evaluation period to judge
overfitting. Each check is also a chance to keep what happened to work there. Prompts that were combined
after looking at evaluation results could not be judged honestly by that same evaluation.

**Rule.** Limit how many times each searcher may look at the evaluation set, record every look, and
report the total with the result.

**Harness.** `check` refuses after `budget.eval_checks_per_agent` looks per agent (default 10), logs every
look (agent, time, prompt, numbers) to `.plab/agents/<agent>/checks.jsonl`, and prints "k/10". Evaluation output is
aggregate only (no per-rule or per-row detail). `report` prints how many times the evaluation set was
looked at, by whom, including the looks made by the selection step itself.

## 5. Decide how to choose before looking, and let the tool choose

**What happened.** The selection rule ("among nominated prompts, those with positive gains on training
and on evaluation and a profitable decision on evaluation; the most profitable one") was written before
the evaluation period was opened, and when nothing qualified, the test period was left unused. That
made "nothing found" a credible result.

**Rule.** Write the selection rule, the splits and the budget down before anyone looks at the
evaluation set. Apply the rule mechanically. Run the test once, on the selected prompt only, and never
again after seeing the result. If nothing qualifies, the test is not run.

**Harness.** `prereg freeze` hashes `prereg.md`, the judging settings of `task.json` (splits, groups,
sampling, decision threshold, selection rule, caps, metric settings, rule wrapper text) and the split
files. `check`, `nominate`, `select` and `test` refuse if any of them changed. `select` re-measures every
nomination and applies the rule; it writes "none" when nothing qualifies. `test --final` refuses without
a selection, and refuses a second run.

## 6. Judge by value, not by ranking quality

**What happened.** Taking an alert won about +2 or lost −50, so the alerts taken had to win more than
96.2% of the time to break even. Several prompts improved AUC without crossing that line. And the
information yardstick does not know the sign: in the synthetic example, a rule whose effect reversed
after the training period still scores near the top of the random distribution on the evaluation set
(the logistic regression simply learns the negative slope), while it loses money.

**Rule.** Report what using the prompt as intended does to the value: the share skipped, the outcome
rate among the cases taken, value per 1,000 cases, next to "take everything".

**Harness.** Every veto prompt gets a decision table at the pre-set threshold (skip when the probability
of skipping is at least 0.5), with value per 1,000 cases and the difference from taking everything.
`fields` prints the break-even rate implied by the mean values. The default selection rule requires a
positive value on the evaluation set. For a task without values, `transfer_gain_pct` measures the
evaluation set in the direction V learned on the training set, so a reversed answer falls to the zero
point there; a selection rule can require it and rank by it.

## 7. Expect the regime to change

**What happened.** Rules about recent trend and volatility worked in both halves of the training period
and in most quarters, then disappeared on the evaluation period, some with the opposite sign. The level
of volatility doubled or tripled, so rules written as absolute levels fired on a very different share of
alerts. What survived, weakly, were narrow rules tied to how the market is built: fixed time windows and
sudden first touches (alerts there lost 1.1 to 1.7 times as often). None was strong enough alone to cross
the break-even line. Re-learning month by month did not help either: about three losses a month is too
little information to learn a new relationship quickly.

**Rule.** Prefer rules with a structural reason over rules fitted to a recent pattern. Check that a rule
holds in every period of the training data, not only on average. Watch how often a rule fires as well as
what it predicts.

**Harness.** `drift` shows, per quarter, half or year of the training split, how often each rule fires,
the outcome rates of skipped and taken cases and the value. `check` prints the skip share on evaluation
next to the training share, so a rule that suddenly fires on many more (or fewer) cases is visible.
Before any search, `survey` measures for each state field, or each rule the machine can answer, how its
slope moves across the periods of the training split and what re-learning it every month would gain
(README.md, section 7); on the FX rows the largest relation that one fixed slope could use gathered
0.5 nats a year, too little to learn quickly, as the case study found.

## What remains a matter of discipline

The harness cannot stop someone from opening the evaluation or test rows directly. It keeps them in
separate files, the brief for searching agents forbids opening them, and every look through the tool is
logged and reported. A study is only as honest as the people and agents running it.
