# prompt-lab

*This is the English translation of [README.md](README.md) (in Japanese), which is the source.*

prompt-lab is a harness that measures, without overfitting, how many more bits about an outcome a decision model can read thanks to a prompt.
This README first explains the idea behind its yardsticks. How to use it starts at "Quick start (no key needed)".
It is written for engineers who know entropy, cross-entropy and mutual information. It skips the definitions and keeps to what is estimated and where the bound breaks down.

A decision model (such as `typesafe-ai/jev` on the Vercel AI Gateway) takes a state and typed questions and returns its answers as probabilities.
For a boolean question it returns the probability of "yes", for a choice a probability per option, and for a score a probability per level.
Because the answers are probabilities, how good a prompt is can be measured in a unit that does not depend on the task: the bits it recovers about the outcome.

## 1. Setting and notation

- `X`: the contents of one row (a case)
- `T`: what the decision model is given to read: `X` itself, or a text made from `X`
- `Z`: the outcome, known later. For an FX alert, "did the trade lose"; for a summary, "the correct answer to a question about the original article"
- `q(z|T)`: the probability the decision model returned, or that probability recalibrated
- `CE`: the mean negative log-likelihood of the true outcomes over the rows measured (bits)
- gain: `H(Z) − CE`. The harness reports it as a percentage of `H(Z)` (`cv_gain_pct`)

The size of `H(Z)` varies a lot between tasks.
On the evaluation split of the FX case study, 54 of 1,270 cases were losses, and `H(Z)` was 0.254 bits. The rarer the outcome, the less room a prompt has to reduce it.

## 2. A lower bound on mutual information

When `q` is a function of `T` alone, the expected cross-entropy splits as follows.

```math
\mathbb{E}\left[-\log_2 q(Z \mid T)\right] = H(Z \mid T) + \mathbb{E}_T\left[ D_{\mathrm{KL}}\big(p(\cdot \mid T) \,\|\, q(\cdot \mid T)\big) \right]
```

Subtracting both sides from `H(Z)` gives:

```math
H(Z) - \mathbb{E}[CE] = I(Z;T) - \mathbb{E}_T\left[ D_{\mathrm{KL}}\big(p(\cdot \mid T) \,\|\, q(\cdot \mid T)\big) \right] \le I(Z;T)
```

The gain is a lower bound on `I(Z;T)`, and its gap to the bound is exactly the expected KL divergence between the decision model's probabilities and the true conditional distribution.
This inequality is known as the variational lower bound (Barber and Agakov, 2003).

Two things follow.

- If the gain measured on new rows is positive, `T` carries at least that much information.
- A small gain does not mean that `T` carries no information. The KL term, the part the decision model failed to read, may simply be large.

Strictly, what is measured is not Shannon's mutual information itself but the information usable within the family V of predictors allowed: the V-information (Xu et al., 2020).

```math
I_V(T \to Z) = H_V(Z \mid \varnothing) - H_V(Z \mid T)
```

`H_V` is the cross-entropy of the best predictor in V.
When V contains the predictors that ignore their input (constants), `H_V(Z|∅)` equals `H(Z)`, and `I_V(T→Z)` is at least 0 and at most `I(Z;T)`.
When V contains every predictor, it equals Shannon's mutual information.

For `H(Z)` the harness uses the value computed from the outcome shares on the rows measured (the plug-in estimate).
In expectation this estimate is at most the true `H(Z)`, so it errs toward underestimating the gain.

## 3. Two ways to measure

### The as-is gain

The probabilities the decision model returned are used as `q` directly.
This amounts to taking V as the single predictor that is the decision model. It contains no constant predictor, so this is not V-information, and it can be negative even in the population.
Besides the information the model failed to read, the KL gap to the bound includes the bias in how it assigns probabilities (poor calibration).
Confident misses weigh heavily. Of 4 statements that are true half the time (`H(Z)` is 1 bit), answer 3 correctly at 0.99 and miss 1 at 0.99: the accuracy is 75%, but the gain is −0.672 bits.

### The recalibrated gain

The decision model's answer `s` (such as the probability of skipping) is recalibrated by a logistic regression of the form `σ(a + b·logit s)` before it is used as `q`.
This is the harness's `cv_gain_pct`.

It estimates the V-information with the predictive family V taken as `{σ(a + b·logit s)}`.
With `b = 0`, V contains the constant predictors, so the population value is at least 0. A negative measured value comes from fitting on finitely many rows, and the zero point of section 4 deals with it.
The quantity estimated is at most `I(Z;s)`, and by the data processing inequality `I(Z;s) ≤ I(Z;T)`.
Recalibration removes uniform shifts and stretches on the logit scale (always saying too high, being overconfident, and so on).

The regression is fitted by cross-validation, and `CE` is measured on the folds not used to fit it. There are 10 folds, repeated 3 times, and by default the rows of one month go into the same fold.
In a task whose outcomes cluster in time, splitting the folds row by row puts rows of the same episode on both sides, and the gain comes out too high.

The recalibrated gain does not care about the sign of the slope `b`, so it is blind to the direction of an answer. A rule whose effect reversed still scores a high gain.
Whether using it gains or loses is checked separately, with the decision table (value per 1,000 cases; item 6 of [The protocol and its reasons](docs/protocol.md)).

### The gain in the training direction

On the evaluation and test splits, the harness also reports a gain (`transfer_gain_pct`) that builds each row's score `η` from the coefficients of V fitted on the training split, and on the split measured refits only the intercept `a` and the strength `c` of `σ(a + c·η)` (`c ≥ 0`).
For an answer whose direction reversed, `c` stops at 0, so the gain falls to the zero point. If the effect only weakened, `c` shrinks and the gain stays positive.
Even in a task without a value field, a selection rule that requires this quantity makes it harder to select an answer whose direction changed after the training split ([ADR-0011](docs/adr/0011-transfer-gain-keeps-the-training-direction.md)).

The difference between the two gains is what you lose by trusting the decision model's probabilities without recalibrating them.
Whether a decision model that claims good calibration really has it can be checked by this difference.

As a check that fits no model, the answers are split into 5 bins of equal weight and the plug-in mutual information with the outcome is reported (`mi_pct`).
A binned plug-in estimate is biased upward, so the value reported subtracts the mean of 500 measurements on shuffled outcomes.

## 4. Estimation bias and the zero point

The bound holds for the expected `CE` measured on rows not used to make `q`. Measuring on finitely many rows brings in two biases.

- The recalibrating regression is fitted on finitely many rows, so even for an answer without information the expected cross-validated gain is negative.
- Measuring on rows used to choose the prompt or the regression adds the optimism of having chosen, and the result is no longer a lower bound.

For the first, random answers, 200 continuous and 200 0/1, are measured with the same yardstick on the same rows, labels, weights and folds. A result is read as its percentile in that distribution.
On the evaluation split of the FX case study (1,270 cases), the median gain of random answers was −0.72%, and the 95th percentile was about 0.
So a result of −0.1% does not show that there is no information. It sits high among random answers.

For the second, the harness uses training, evaluation and test splits divided by time, a cap and a log on looks at the evaluation split, a pre-registration, and a single test run ([The protocol and its reasons](docs/protocol.md)).

## 5. Tasks that produce text

The same yardsticks work for tasks where an LLM writes text, such as summarization. A decision model is placed after the writing LLM and reads only the text written.

```
row X → writing LLM (the prompt being tuned) → text T → decision model (fixed questions) → probabilities → gain
```

With `s` the decision model's answer, the data processing inequality orders the quantities of the stages as follows.

```math
H(Z) - CE \le I(Z;s) \le I(Z;T) \le I(Z;X)
```

Improving the writer's prompt amounts to increasing what the decision model can read from `T`.
In Shannon's terms, no way of writing can exceed `I(Z;X)`.
The decision model and the questions are fixed and written in the pre-registration. If the reading side moved too, its effect could not be told apart from how good the writing side is.

### A summarization example

`Z` is whether a statement about an article is true. Half the statements are true, and half are false statements with exactly one fact changed (`H(Z)` is 1 bit).
The decision model is given different things to read, and the probabilities it puts on "true" are compared. The table is an example with 4 statements for one article.

| what it reads | the 2 true statements | the 2 false statements | CE (bits) | gain (bits) |
|---|---|---|---|---|
| nothing | 0.5, 0.5 | 0.5, 0.5 | 1.000 | 0.000 |
| the summary | 0.8, 0.6 | 0.2, 0.4 | 0.529 | 0.471 |
| the full article | 0.95, 0.9 | 0.05, 0.1 | 0.113 | 0.887 |

Taking reading nothing and reading the full article as the two ends, 53% of what this decision model can read from the full article survives in the summary (`(1.000 − 0.529) ÷ (1.000 − 0.113)`).
This share is a quantity fixed by the decision model and the set of questions.

The share can, however, exceed 100%.
The ordering of the inequalities above holds for Shannon's quantities, and V-information can grow under further processing (Xu et al., 2020).
If the writing LLM rewrites the information in a form the decision model reads more easily, the summary can yield more than the full article.
That can happen, for example, when the full article exceeds the decision model's context length (32,000 tokens for Jev). Read the full-article value not as an upper bound but as a reference mark on the scale.

### Writing the questions, and the screens

An LLM writes the questions for each article, and the decision model screens them twice.

- Questions it misses when reading the full article are dropped. This removes questions whose correct answer was set wrong.
- Questions it gets right when reading nothing are dropped. This removes questions answerable from the question text and what the decision model remembers.

Strictly, the decision model also reads the question text `Q`, so what is measured is a lower bound on `I(Z; T, Q)`.
The gain when reading nothing stands for `I(Z;Q)`. With the second screen keeping it near 0, the rest can be read as what `T` carried.

These two screens also give the two ends of the table above.
The question-writing LLM is not shown the summaries, and how the questions are written goes in the pre-registration.
The questions are split into training and evaluation sets too, and cross-validation keeps the questions of one article in the same fold, because questions about the same article are not independent.

## 6. Tasks it fits

| task | what is tuned | `Z` |
|---|---|---|
| judging, classifying, skipping (FX veto rules, for example) | the prompt given to the decision model | the correct label |
| summarizing, extracting, reporting | the writing LLM's prompt | answers to questions about the source document |
| comparing decision models | the decision model (Jev, Gemini Decision, Liquid d1, and so on) | the correct answers for the same rows and questions |

When comparing decision models, compare the usable information by the recalibrated gain, and the calibration by its difference from the as-is gain.
Choosing which model to use is itself a kind of fitting, so the comparison is made on the training rows, and it also counts toward the looks at the evaluation rows.

## 7. Information that holds across time

In this section `t` is time and `F` a factor. They are different symbols from the `T` of sections 1–6 (what the decision model reads).
A factor is something that gives one number per row: the value of a state field, whether a text field takes a given value, or a rule or question of a prompt as the machine (`expr`) answers it.
`plab survey` measures, factor by factor, whether its relation to the outcome holds across periods. It is used before starting a new study, to check whether the data hold information worth searching for.

### 7.1 The question

A relation found on the training split is used in the periods after it. If the relation moves from period to period, the information usable with one fixed relation is less than what knowing each period's relation would give.
This section is about the size of that difference and what remains for a fixed relation.

### 7.2 The split

```math
I(Z;F \mid t) = I(Z;F) + \big[\, I(Z;F \mid t) - I(Z;F) \,\big], \qquad I(Z;F \mid t) - I(Z;F) = I(Z;t \mid F) - I(Z;t)
```

The left side is the information usable when the relation at each time is known, and the first term is the information usable with one relation over all periods.
The bracket holds the information that rides on the relation changing with time. When `F` and `Z` are independent given `t`, the bracket is `−I(Z;F)`, which is negative.
A search that writes fixed rules reaches the first term at most. Using the bracket requires re-learning the relation while using it.
`plab survey` measures this split in a form that follows each period's outcome share but fixes the slope (7.3).

### 7.3 Forecasting with a stale coefficient

With `p_t(z|F)` the true conditional distribution at time `t`, forecasting with a `q(z|F)` fitted at another time gives the following gain against that period's outcome share.

```math
\mathrm{gain}(q) = I_t - \mathbb{E}_F\, D_{\mathrm{KL}}\big(p_t(\cdot \mid F) \,\|\, q(\cdot \mid F)\big), \qquad I_t = I(Z;F \mid t)
```

Standardize `F` to mean 0 and standard deviation 1, and let `π` be the outcome share. For a logistic V with a small slope `b`, `p(F) − π ≈ π(1 − π) b F` and `KL ≈ (p − π)² / (2π(1 − π))` give the approximation:

```math
I_t \approx \kappa\, b_t^2, \qquad \kappa = \frac{\pi(1-\pi)}{2}\ \text{nats}
```

With the intercept refitted and the slope fixed at `b`, forecasting at time `t` leaves a KL term of about `κ(b − b_t)²`. The gain of the fixed slope is `κ(2 b b_t − b²)`.
With `μ` the mean of the true per-period slopes and `σ²` their variance between periods, the gain averaged over periods is largest at `b = μ`, where it is `κμ²`. Knowing each period's slope would give `κ(μ² + σ²)`.

```math
\underbrace{\kappa(\mu^2+\sigma^2)}_{\text{each period's slope known}} = \underbrace{\kappa\mu^2}_{\text{usable with one slope}} + \underbrace{\kappa\sigma^2}_{\text{rides on the slope's changes}}
```

The tables write the share usable with one slope, `μ² / (μ² + σ²)`, as `share`.
With several answer columns, read `b` as a vector measured in the covariance of the columns. With `θ` the angle between the learned coefficients and the current ones, `I_t cos²θ` remains and `I_t sin²θ` is lost.
`transfer_gain_pct` (section 3) refits a strength `c ≥ 0`, so it stops at 0 when `cos θ < 0`. With one column, `θ` is 0 or π, so as long as the strength is refitted, a fixed factor loses only when its direction reverses.

At an outcome share of 5%, `κb²` differs from the exact `I(Z;F)`, computed by Gauss-Hermite quadrature, by 0.7% at `b = 0.3` and by 2.1% at `b = 0.5`.

### 7.4 The estimate per factor

- A factor's values are turned into midranks (ties get the mean of their ranks) and then standardized to mean 0 and standard deviation 1. A row without a value sits at the mean rank. A heavy-tailed field and a 0/1 field compare on the same `κ`, and a monotone change of a field's unit leaves the numbers unchanged. The rows are in time order, so ranking ties by row position would put a time trend into the factor. Midranks prevent that.
- A state field that is a number or a boolean becomes one factor. A text field becomes one 0/1 factor per value, only when it has at most 12 values, and only for values held by 2–98% of the rows. Given a prompt, the factors are each rule, each question that combines two or more rules, and each question with an `expr` (a choice gives one factor per option, a score its level index). Rules and questions without an `expr` are listed as not surveyed, with the reason.
- In every period (`--by`, half a year by default), a one-factor logistic regression `logit P = a + b F` is fitted, giving the slope `b_t` and its standard error `s_t` from the Fisher information. It reads every row of the split at weight 1, not the down-sampled training sample: a factor's values cost nothing, so there is no reason to down-sample. A period with fewer than 5 rows of either class is left out.
- The summary across periods is the DerSimonian-Laird random-effects estimate. `μ` is the mean weighted by `1/(s_t² + σ̂²)`, and its standard error is `SE(μ) = (Σ 1/(s_t² + σ̂²))^{-1/2}`.

```math
w_t = s_t^{-2},\qquad Q = \sum_t w_t\,(b_t - \bar b_w)^2,\qquad \hat\sigma^2 = \max\!\Big(0,\ \frac{Q - (k-1)}{\sum_t w_t - \sum_t w_t^2 / \sum_t w_t}\Big)
```

- `μ²` as it stands comes out larger by `SE(μ)²`, so the survey uses `μ₂ = max(0, μ² − SE(μ)²)`. This is the same treatment as subtracting the per-period sampling noise to get `σ̂²`. In the tables, `info%` is `100 κ μ₂ / H(Z)` (`H(Z)` in nats), `share` is `μ₂ / (μ₂ + σ̂²)`, and `nats/yr` is a year's number of rows times `κμ₂`.
- `σ̂` is truncated at 0, so it is biased upward near 0. In the synthetic runs of 7.7, the mean of `σ̂` was 0.07 when the true `σ` was 0.
- The expected value of `Q` is `(k − 1) + cσ²` (`c` is the denominator in the formula above). The `σ` at which `Q` reaches the threshold of 7.6 is reported as the floor. A `σ` below the floor cannot be told from 0 with that number of periods.
- When the outcome takes three or more values, the same is done for each value other than the reference, on the rows of that value and the reference only. As in V, each value is compared against the reference, and a table is printed per value. With two values it is done once.

### 7.5 What can be learned, and the gain of re-learning

The expected KL of a maximum-likelihood fit is `d/(2N)` for `d` fitted numbers, so with `N` rows in a window, fitting one slope costs about `1/(2N)` nats per forecast row.
The gain per row is about `I − 1/(2N)`, and the usable share is `1 − 1/(2NI)`. If the window gathers `N·I` = 1 nat of information, half remains; at 5 nats, 90%.

When the slope moves as a random walk, what can be tracked is limited too. If the slope takes `τ` periods to move by its own size, with `n` rows a period and `I` nats of information per row, the share usable by tracking is about:

```math
1 - \frac{1}{\sqrt{2 n \tau I}}
```

One period's rows carry a Fisher information of `2κn` about the slope. With a step variance of `b²/τ`, the steady-state error variance `P` of a Kalman filter is `b/√(2κnτ)`, and each row loses `κP`. Dividing `κP` by `I = κb²` gives the second term of the formula above.

`plab survey` measures the gain of re-learning (`pq` in the tables) directly. Following Dawid's (1984) prequential approach, each month is forecast by `a + bF` fitted on a window before that month.
The forecast is compared with one that re-learns only the outcome share on the same window, in % of the `H(Z)` of the rows forecast. The windows are 6, 12 and 24 months, and all earlier months (`all`). `--gap` takes the number of months until an outcome is known, and the window ends that many months earlier.

- Every window forecasts the same months. The months forecast are those with the longest fixed window plus `--gap` months before them, and with rows of both classes in every window's months. Forecasting different months per window would compare which window wins on different targets.
- A fixed window is used only when it is at most half the months of the surveyed span. With a training split of two and a half years, the 24-month window is not used.
- The standard error comes from the spread of the monthly gains. Outcomes cluster in time, so months, not rows, are the units. Cells beyond 2 standard errors get `*`.
- The `all` window roughly measures a fixed relation, and the short windows a tracked one.
- Where a factor without information lands is given per window by the formula below. `n_m` is the number of rows in month m, `N_m` the number of rows in that month's window, and `n` the number of rows forecast.

```math
-100\ \frac{\sum_m n_m / (2 N_m)}{n\, H(Z)}
```

In the synthetic runs of 7.7, the 24-month and all-month windows landed where this formula says, and the shorter the window, the slightly lower the measured position (0.02–0.09 points on the 6-month window). With only about 30 positives in a window, learning probably costs more than `1/(2N)`.

### 7.6 Reading

Each factor gets two tests and is read as one of four words. The mean counts when `|μ| ≥ 2 SE(μ)`. The spread counts when `Q`, turned into a standard normal (the Wilson-Hilferty approximation), is 2 or more.

| | spread does not count | spread counts |
|---|---|---|
| mean counts | `stable` | `partly` |
| mean does not count | `none` | `moving` |

A factor with fewer than 4 usable periods is `few`.

- The chance that a factor without information looks like something on either test is about 6.7% (from 4.6% two-sided and 2.3% one-sided). The summary prints the number of factors times this rate, as the count expected by chance. Nothing is corrected for multiple comparisons.
- The `μ` of a rule found by searching the same rows comes out larger by the amount of choosing. Either measure only rules with a reason before looking at the rows, or discount them.
- `stable` means that within the surveyed periods the slope did not move by more than the floor. It says nothing about later periods. Surveying the training split alone, a rule whose direction turns right after the training split also looks `stable`, as rule (b) of the synthetic example in 7.7 does.
- For a strong factor, the weak-effect approximation breaks down. At an outcome share of 5%, `κb²` overstates the exact quantity by 13% at `|b| = 1`, and at a share of 40% by 32%. Read strong factors by the `pq` columns. On the synthetic three-class rows (`ticket_rows` in `tests/helpers.py`), a factor with `μ` of 2 had an `info%` of 70%, and its re-learning gain was 49%.

The tables fall into three shapes. Information that remains reads `stable` or `partly` with a large `share`, and has `pq 24` and `pq all` above the no-information position. Information riding on change has `σ` at least `|μ|`, and its short windows do no worse than its long ones. Without information, about as many factors read as something as chance gives, and every window sits near the no-information position.
Whether to start a search is decided by combining these numbers with things outside the data (how the rules were made, whether new rows will keep coming). `plab survey` does not decide.

### 7.7 Numbers checked

These are synthetic numbers: 95 rows a month, an outcome share of about 5% (intercept −3), and a true slope that is AR(1) over half-years. They were run 40 times with different seeds (built like `planted` in `tests/test_survey.py`).

| planted slope | readings, 20 half-years | readings, 8 half-years | `μ` over 20 half-years (`SE`) | `σ̂` over 20 half-years (floor) |
|---|---|---|---|---|
| constant 0.3 | stable 40 | stable 39, none 1 | 0.286 (0.049) | 0.072 (0.174) |
| mean 0.3, spread 0.3 | partly 37, stable 2, moving 1 | stable 15, partly 13, moving 11, none 1 | 0.308 (0.080) | 0.288 (0.175) |
| spread 0.4 around 0 | moving 34, partly 6 | moving 25, none 9, partly 4, stable 2 | 0.014 (0.095) | 0.369 (0.175) |
| no information | none 37, stable 2, moving 1 | none 40 | −0.002 (0.048) | 0.057 (0.173) |

The mean gain of re-learning over 20 half-years (% of `H(Z)`):

| planted slope | 6 months | 12 months | 24 months | all months |
|---|---|---|---|---|
| constant 0.3 | +0.47 | +0.73 | +0.87 | +0.93 |
| spread 0.4 around 0 | +0.58 | +0.48 | +0.20 | −0.12 |
| no information | −0.48 | −0.26 | −0.11 | −0.04 |
| no-information position (formula) | −0.46 | −0.23 | −0.12 | −0.05 |

- With 20 half-years, the readings come out as planted. A factor without information looked like something 3 times in 40, in line with the expected 6.7%.
- With 8 half-years (a four-year training split), the floor of the spread is 0.23. A spread of 0.3 is only a little above it, so 15 of the 40 runs read `stable`. The floor tells you this in advance.
- A constant slope gains most by re-learning on all months, and a moving slope gains more the shorter the window. For the factor without information, 0 of 160 cells got `*`, over 20 half-years and over 8.

In the synthetic example of the quick start (seed 7), `walkthrough.py` also runs the survey. Rule (b) (skip when the 24-hour move is strong) changes direction in 2022.
Surveying the training split (2018–2021) alone, both (a) and (b) read `stable`. (b) has `μ = +0.26` and a `pq 12` of +1.69% *.
Surveying every split (2018–2025) after the test, (a) stays `stable` (`μ = +0.26`, `share` 0.89, `pq all` +0.93% *). (b) becomes `moving` (`μ = +0.10`, `σ̂ = 0.20`, `share` 0.11, `pq all` −0.12%).

On the FX case study's rows (January 2022 to October 2026, 5,170 cases, 242 losses), a study surveyed every split after its test run, with these results.

- Of the 47 factors made from the state fields, 3 read `stable`, 3 `moving` and 41 `none`. Chance alone makes 3.2 look like something.
- Even the factor with the most information usable by one slope has 0.24% of `H(Z)`, which gathers only 0.5 nats a year. Even the 24-month window holds about 1 nat, and the cost of learning takes half of it.
- No re-learning cell got `*` (the largest was +0.35%, with a standard error of 0.25). The floor of the spread is 0.23, and smaller movements are not visible.
- Compared with the prototype's numbers on rows from September 2016, `μ` fell within 2 standard errors for 31 of the 32 number and boolean fields. The 1 outside and the 2 near it were fields for the level of volatility. Over 2016–2026 they had a negative slope, but on the rows from 2022 it is about 0.

## 8. What is built and what is still a plan

The yardsticks are in `src/yardsticks.py`. What is built:

- The as-is gain (`asis_gain_pct`). It is reported when the prompt's `forecast` says which option of which question forecasts which outcome.
- The gain in the training direction (`transfer_gain_pct`). It is reported on the evaluation and test splits.
- The recalibrated gain (`cv_gain_pct`). It also handles outcomes with three or more values and rows with several questions. V is the family that gives the probability of each outcome value by softmax from features lining up the answers of every question (the log ratio of each option's probability to that of the reference option). With two outcome values and one question it gives the same number as the `σ(a + b·logit s)` of section 3.
- Mutual information over 5 bins, minus its mean on shuffled outcomes (`mi_pct`). It is reported only for two outcome values and a prompt that writes a `forecast`.
- The zero point measured with random answers, per answer shape (the kind of each question and its number of options). The random answers are of two kinds: ones that spread their probability (soft) and ones that put it on one option (sharp).
- Calls to the decision model, through `/v1/decisions` of the Vercel AI Gateway (`src/gateway.py`).
- The survey of how each factor's coefficient changes over time, from section 7 (`plab survey`, `src/survey.py`). It calls no decision model and reads only the state fields and the machine's answers.

Two parts are still at the plan stage:

- the writing-LLM stage (section 5)
- the stage that writes questions and screens them

[日本語の README](README.md) · [The protocol and its reasons](docs/protocol.md) · [Design decisions (ADR)](docs/adr/README.md)

Requirements: Python 3.10+ and numpy. No other dependency.

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
the `fake` decider (made-up answers for tests). What you will see (seed 7):

| prompt (machine decider) | training: cv_gain (percentile among random answers), AUC halves | evaluation: AUC, value per 1,000 vs take-everything |
|---|---|---|
| (b) skip when `move_24h_toward > 12` | 0.69% (100th), 0.528 / 0.571, lift > 1 in every year | 0.464 (3rd percentile), −78; skips 12% → 29% because volatility grew |
| (b) relative to volatility | | cv_gain 0.54% (100th) but AUC 0.460, −140: information with the wrong sign |
| (a) skip 13:00–15:59 UTC | 0.52% (100th), 0.562 / 0.553 | 0.565, cv_gain 0.56% (100th), +356 |
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

A full evaluation (1,000–1,300 calls of about 600 tokens each) costs about $0.03–0.04.

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
[plab-survey](.claude/skills/plab-survey/SKILL.md) (in Japanese) runs `plab survey` and decides whether a
search is worth starting.

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
| `survey [PROMPT] [--by quarter\|half\|year] [--gap MONTHS] [--splits train\|all]` | train (`all`: every split) | the state fields, or a prompt's rules and questions as the machine answers them, as factors: each one's slope by period, its mean and spread, and the gain of re-learning it (section 7). Writes nothing on the training split; `--splits all` runs only on a closed study, is marked EXPLORATORY and is logged to `surveys.jsonl` |
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
  - `"fitted"`: the forecast is V fitted on the training answers. There is no as-is gain. On the training
    split it is out of fold, so its AUC is taken within each fold and averaged, and binned MI, the
    quintiles and the cross-fit, which compare units of different folds, are left out. Pooled over folds,
    a weak answer's AUC fell below 0.5 and its MI rose.
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

`expr` is a small, safe language (parsed with `ast`, never `eval`): field names, numbers, strings,
`True/False/None`, tuples/lists, comparisons (`== != < <= > >= in, not in`, chained), `and/or/not`,
`+ - * /`, `abs/min/max/len`, and subscripts/slices. An `expr` may only read fields that the prompt gives
to the decider. If a field it reads is null (or a list element it uses is null), or the computation fails
(type mismatch, division by zero), the expr has no value: a rule does not apply, a predicate says no, and a
choice or score counts as refused.

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
except in a down-sampled training split). Sections 1–4 of this README give the reasons.

The answers, encoded. An answer is a probability over a question's options. A predicate gives
logit(P(yes)), a choice gives log(p_j / p_0) for each option after the first, and a score gives one
column, logit(mean level / (levels − 1)). Probabilities are clipped to [1e-4, 1 − 1e-4] first. A choice
answered without probabilities is a point mass on the chosen option. A refused question counts as the
uniform answer: its unit stays in, and the record counts it in `refused`.

Information, for any K and any number of questions:

- `cv_gain_pct`, the recalibrated gain: V is a multinomial logistic regression (softmax, class 0 as
  reference) on the encoded answers of every question, fitted out of fold (10 folds grouped by
  `cv_groups`, 3 repeats, seeds 100 + r, a 1e-3 ridge on the slopes only). The result is
  100 · (H(Z) − cross-entropy) / H(Z). It estimates the V-information of the answers about the outcome.
  With one predicate and two classes it is exactly the number 0.1.0 gave. **Sign-blind.**
- `transfer_gain_pct`, the gain in the training direction (evaluation and test splits): V is fitted on
  all training answers, and its coefficients without the intercepts give each case a score E per class.
  On the split measured, E stays fixed and only the intercepts and one strength c ≥ 0 are fitted, out of
  fold with the folds of cv_gain. An answer whose direction reversed falls to the zero point; one whose
  effect only weakened keeps a positive gain. Whatever the number of answer columns, it fits only one
  number more than the constant forecast. Measuring evaluation or test also uses the training answers
  (asking the decider for those not stored, and counting the rows left without an answer in the record's
  `trained_on`). Sign-aware.
- `asis_gain_pct`, the as-is gain: the same share for the decider's own probabilities, through the
  prompt's `forecast`. Nothing is fitted, so it can be below zero on any number of units;
  cv_gain − asis_gain is what trusting the raw probabilities without recalibrating them costs. Sign-aware.

Use, for two classes and a forecast (s is the forecast probability of class 1):

- AUC (weighted, ties count half), and the AUC on each training half. Sign-aware.
- `mi_pct`, binned MI: s in 5 bins of equal weight, the mutual information with Z minus the mean of 500
  shuffles, in % of H(Z), with its permutation p. It fits no model. **Sign-blind.**
- The positive rate by forecast quintile.
- Decision (with `decision` and values): skip when s ≥ the threshold. The skip share, the positive and
  negative rates among the cases taken, the positive rate among those skipped and the lift, the value per
  1,000 cases, and the same for taking everything. Sign-aware.
- Cross-fit (training, with values and `half_boundary`): the best skip share (5% steps) chosen on one
  half and applied to the other, both ways. Sign-aware.
- Test: cases taken, positive and negative rates with Wilson 95% intervals, value per case and in
  total, AUC with a one-sided permutation p, and binned MI; next to taking everything.

Information that holds across time (`plab survey`). Section 7 defines the columns `mu`, `sigma`,
`share`, `info%`, `nats/yr` and `pq` of its tables, and says how to read them.

Zero point. The same yardsticks on random answers of the prompt's shape (the kind and number of
options of every question and, with two classes, which options forecast class 1), keeping the split's
labels, weights and groups. There are two kinds, 200 of each: *soft* answers spread their probability at
random, and *sharp* answers put all of it on one option. A result is placed among the sharp kind when at
least 90% of its weight has at most 0.15 of its probability off one option on every question, and among
the soft kind otherwise. Every record prints its percentile. On evaluation and test, the direction of
`transfer_gain_pct` comes from V fitted on other random answers.

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

## References

- D. Barber and F. Agakov. The IM Algorithm: A Variational Approach to Information Maximization. NIPS 2003.
- Y. Xu, S. Zhao, J. Song, R. Stewart, and S. Ermon. A Theory of Usable Information under Computational Constraints. ICLR 2020.
- T. M. Cover and J. A. Thomas. Elements of Information Theory, 2nd ed. Wiley, 2006.
