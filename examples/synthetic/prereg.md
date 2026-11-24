# Pre-registration: synthetic alerts (walkthrough)

Written before anyone looked at the evaluation split. The data is synthetic (`make_data.py`).

## Question
Each row is an alert. Taking it wins 2 (label 0) or loses 50 (label 1). A veto prompt says
"skip this alert". Success = the alerts that are taken are worth more than taking every alert,
on data the search never saw.

## Splits (task.json)
- training 2018-01-01 to 2021-12-31: searchers look freely (halves: before / from 2020-01-01);
- evaluation 2022-01-01 to 2023-12-31: only through `plab check`, at most 4 looks per agent;
- test 2024-01-01 to 2025-12-31: once, for the selected prompt, through `plab test --final`.

## Selection rule (task.json "selection")
Among the nominated prompts, keep those with
training cv_gain > 0, evaluation cv_gain > 0, evaluation cv_gain at or above the 90th percentile
of random answers, and evaluation value per 1,000 alerts above "take everything".
Choose the one with the largest gain over "take everything" on the evaluation split (ties: fewer
skips). If none qualifies, the test split stays unused and the result is "nothing found".

## Decision
Skip when the answer (probability of skipping) is 0.5 or more.

## Budget
$0.50 of model calls (the walkthrough uses the free machine and fake deciders), 4 evaluation checks
and 2 nominations per agent.

## What will be reported
The test table for the selected prompt next to "take everything", the number of looks at the
evaluation split, and the cost.
