"""promptlab.survey: the arithmetic on planted truths and on formulas worked out by hand, and the
command through `plab`. No expected number here comes from a run of the code under test."""
import json
import math
import unittest

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from promptlab import survey as S
from tests.helpers import TempTask, alert_rows, alert_task, ticket_rows, ticket_task


def planted(mu, sigma, rho, seed, half_years, per_month=95, base=-3.0):
    """A factor x ~ N(0, 1) whose true slope follows an AR(1) over half-years around mu, with spread
    sigma: logit P(y = 1) = base + b_h x. Returns x, y, month and period."""
    rng = np.random.default_rng(seed)
    b = np.empty(half_years)
    b[0] = mu + sigma * rng.normal()
    for i in range(1, half_years):
        b[i] = mu + rho * (b[i - 1] - mu) + sigma * math.sqrt(1 - rho ** 2) * rng.normal()
    half = np.repeat(np.arange(half_years), per_month * 6)
    month = half * 6 + rng.integers(0, 6, len(half))
    x = rng.normal(size=len(half))
    y = (rng.random(len(half)) < 1 / (1 + np.exp(-(base + b[half] * x)))).astype(int)
    return x, y, month, np.array([f"{h:03d}" for h in half])


def path(bs, se):
    return [S.PeriodSlope(f"p{k}", 100, 10, b, se) for k, b in enumerate(bs)]


STABLE, PARTLY, MOVING, NONE = (0.3, 0.0, 0.0), (0.3, 0.3, 0.5), (0.0, 0.4, 0.5), (0.0, 0.0, 0.0)


class Arithmetic(unittest.TestCase):
    def test_midranks_share_ties_and_do_not_depend_on_position(self):
        got = S.standardize(np.array([3.0, 1.0, 3.0, 2.0, np.nan]))
        rank = np.array([3.5, 1.0, 3.5, 2.0, 2.5])           # ties share 3.5; the unit without a value sits at the mean rank
        np.testing.assert_allclose(got, (rank - 2.5) / math.sqrt(0.9))
        x = np.random.default_rng(1).integers(0, 4, 400).astype(float)
        perm = np.random.default_rng(2).permutation(400)
        np.testing.assert_array_equal(S.standardize(x)[perm], S.standardize(x[perm]))
        np.testing.assert_allclose(S.standardize(np.exp(x)), S.standardize(x))
        np.testing.assert_allclose(S.standardize(-x), -S.standardize(x))
        self.assertIsNone(S.standardize(np.array([2.0, 2.0, np.nan])))

    def test_shuffling_tied_units_changes_no_number(self):
        """A factor with ties, in time order. Swapping the order of units that share a value and a month
        leaves every number of the survey as it was."""
        x, y, month, period = planted(*STABLE, seed=4, half_years=8)
        x = np.round(x)
        order = np.lexsort((np.random.default_rng(3).random(len(x)), x, month))
        vs = S.Versus("1", "0", np.ones(len(y), bool), y, month, period)
        moved = S.Versus("1", "0", np.ones(len(y), bool), y[order], month[order], period[order])
        windows = (6, 12, 24, None)
        a = S.measure(S.Factor("x", "field", x, 100.0), vs, S.walk(month, y, windows, 0), 1000.0)
        b = S.measure(S.Factor("x", "field", x[order], 100.0), moved, S.walk(month[order], y[order], windows, 0), 1000.0)
        for key in ("mu", "mu_se", "sigma", "q", "info_pooled_pct"):
            self.assertAlmostEqual(a[key], b[key], delta=1e-3, msg=key)
        for key, r in a["relearning"].items():
            self.assertAlmostEqual(r["gain_pct"], b["relearning"][key]["gain_pct"], delta=1e-3, msg=key)

    def test_random_effects_with_equal_errors_are_the_hand_formula(self):
        """With one standard error s for every period: mu = mean(b), tau2 = var(b) - s^2 (0 when below),
        se(mu) = sqrt((s^2 + tau2) / k)."""
        for bs, s in (([0.1, 0.5, -0.2, 0.4, 0.3], 0.1), ([0.1, 0.3, 0.2, 0.2], 0.1)):
            ap = S.across_periods(path(bs, s))
            tau2 = max(0.0, np.var(bs, ddof=1) - s ** 2)
            self.assertAlmostEqual(ap.mu, np.mean(bs), places=12)
            self.assertAlmostEqual(ap.tau2, tau2, places=12)
            self.assertAlmostEqual(ap.mu_se, math.sqrt((s ** 2 + tau2) / len(bs)), places=12)
        self.assertIsNone(S.across_periods(path([0.1, 0.2, 0.3], 0.1)))

    def test_the_sampling_noise_is_taken_out_of_mu_squared(self):
        """b = 0.1, 0.3, 0.2, 0.2 with s = 0.1: tau2 = 0, se(mu) = 0.05, mu2 = 0.2^2 - 0.05^2. With s = 0.2 around
        0.05, se(mu) = 0.1 is above |mu|, so mu2 = 0 and one fixed slope is worth nothing."""
        ap = S.across_periods(path([0.1, 0.3, 0.2, 0.2], 0.1))
        self.assertAlmostEqual(ap.mu2, 0.04 - 0.0025, places=12)
        self.assertEqual((ap.reading, ap.stable_share), ("stable", 1.0))
        weak = S.across_periods(path([0.0, 0.1, 0.05, 0.05], 0.2))
        self.assertEqual(weak.mu2, 0.0)
        self.assertEqual((weak.reading, weak.stable_share), ("none", None))
        spread = S.across_periods(path([0.1, 0.5, -0.2, 0.4, 0.3], 0.1))
        mu2 = max(0.0, 0.22 ** 2 - spread.mu_se ** 2)
        self.assertAlmostEqual(spread.stable_share, mu2 / (mu2 + np.var([0.1, 0.5, -0.2, 0.4, 0.3], ddof=1) - 0.01), places=12)

    def test_the_spread_test_and_the_floor_of_sigma_follow_the_chi_square(self):
        """Without spread, q is chi-square with k - 1 degrees; its point exceeded with the one-sided normal
        probability at Z_CUT is where the spread starts to count, and the floor of sigma is the tau2 at which q is
        expected there: E[q] = k - 1 + c tau2, c = (k - 1) / s^2 for equal errors."""
        df, s = 7, 0.2
        upper = math.erfc(S.Z_CUT / math.sqrt(2)) / 2
        q_cut = float(np.quantile(np.random.default_rng(11).chisquare(df, 400_000), 1 - upper))
        ap = S.across_periods(path(np.random.default_rng(12).normal(0.1, s, df + 1), s))
        self.assertAlmostEqual(ap.sigma_floor, math.sqrt(s ** 2 * (q_cut - df) / df), delta=0.01)
        self.assertAlmostEqual(S._chi2_z(q_cut, df), S.Z_CUT, delta=0.05)

    def test_kappa_is_the_weak_effect_mutual_information(self):
        """x ~ N(0, 1), logit P = a + b x near a 5% rate: kappa b^2 against I(Z; x) by Gauss-Hermite quadrature."""
        nodes, weights = hermegauss(80)
        weights = weights / math.sqrt(2 * math.pi)
        for b, within in ((0.1, 0.002), (0.2, 0.005), (0.3, 0.01), (0.5, 0.03)):
            p = 1 / (1 + np.exp(-(-2.944 + b * nodes)))
            pi = float(weights @ p)
            exact = float(weights @ (p * np.log(p / pi) + (1 - p) * np.log((1 - p) / (1 - pi))))
            self.assertAlmostEqual(S.kappa(pi) * b ** 2 / exact, 1, delta=within)

    def test_the_reading_of_planted_slopes(self):
        """20 half-years, 95 units a month, 5% positive, 10 seeds each."""
        for planted_as, (mu, sigma, rho) in (("stable", STABLE), ("partly", PARTLY), ("moving", MOVING), ("none", NONE)):
            found = []
            for seed in range(10):
                x, y, _, period = planted(mu, sigma, rho, seed, 20)
                found.append(S.across_periods(S.period_slopes(S.standardize(x), y, period)))
            readings = [ap.reading for ap in found]
            self.assertGreaterEqual(readings.count(planted_as), 8, (planted_as, readings))
            if planted_as == "stable":
                self.assertAlmostEqual(np.mean([ap.mu for ap in found]), 0.3, delta=0.05)
            if planted_as == "moving":
                self.assertAlmostEqual(np.mean([ap.sigma for ap in found]), 0.4, delta=0.1)

    def test_a_thin_period_is_left_out_and_too_few_periods_read_few(self):
        x, y, _, period = planted(*STABLE, seed=1, half_years=6)
        thin = (period == "002") & (y == 1)
        keep = ~thin | (np.cumsum(thin) <= S.MIN_PER_CLASS - 1)
        found = S.period_slopes(S.standardize(x[keep]), y[keep], period[keep])
        self.assertEqual([p.period for p in found], ["000", "001", "003", "004", "005"])
        self.assertIsNone(S.across_periods(found[:3]))

    def test_every_window_forecasts_the_same_months_and_never_sees_them(self):
        """Months 0..39 with 20 units each, gap 2, windows 6, 12 and all: the first forecast is month 12 + 2,
        and month 30, whose 6-month window holds no positive, is forecast by no window."""
        month = np.repeat(np.arange(40), 20)
        y = np.zeros(len(month), int)
        y[::7] = 1
        y[(month >= 22) & (month < 28)] = 0
        w = S.walk(month, y, (6, 12, None), 2)
        self.assertEqual(w.months, tuple(m for m in range(14, 40) if m != 30))
        for steps in zip(*w.steps.values()):
            forecasts = [at for _, at in steps]
            for at in forecasts[1:]:
                np.testing.assert_array_equal(at, forecasts[0])
            m = int(month[forecasts[0]][0])
            fitted = [sorted(set(month[fit].tolist())) for fit, _ in steps]
            self.assertEqual(fitted[0], list(range(m - 2 - 6, m - 2)))
            self.assertEqual(fitted[1], list(range(m - 2 - 12, m - 2)))
            self.assertEqual(fitted[2], list(range(0, m - 2)))

    def test_the_no_information_position_is_the_hand_formula(self):
        """Two forecast months of 20 units after 12 months of 20: the window of 6 fits on 120 units, the one of
        all months on 240 and 260. The cost is n / (2N) nats per month, in % of H(Z) of the forecast units."""
        month = np.repeat(np.arange(14), 20)
        y = np.zeros(len(month), int)
        y[::4] = 1
        w = S.walk(month, y, (6, None), 0)
        self.assertEqual(w.months, (6, 7, 8, 9, 10, 11, 12, 13))
        w = S.Walk(w.months[-2:], {k: v[-2:] for k, v in w.steps.items()})
        h_nats = 40 * -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
        got = S.no_information_pct(y, w)
        self.assertAlmostEqual(got["6"], -100 * (20 / 240 + 20 / 240) / h_nats, places=12)
        self.assertAlmostEqual(got["all"], -100 * (20 / 480 + 20 / 520) / h_nats, places=12)


class Relearning(unittest.TestCase):
    """12 half-years (72 months, so the windows of 6, 12 and 24 months all fit), 6 seeds of each slope."""

    @classmethod
    def setUpClass(cls):
        cls.found = {}
        for name, truth in (("stable", STABLE), ("moving", MOVING), ("none", NONE)):
            gains, zero = [], []
            for seed in range(6):
                x, y, month, _ = planted(*truth, seed=100 + seed, half_years=12)
                w = S.walk(month, y, S.WINDOWS, 0)
                gains.append({k: r.gain_pct for k, r in S.relearning(S.standardize(x), y, w).items()})
                zero.append(S.no_information_pct(y, w))
            cls.found[name] = ({k: np.mean([g[k] for g in gains]) for k in gains[0]}, {k: np.mean([z[k] for z in zero]) for k in zero[0]})

    def test_a_factor_without_information_lands_near_the_analytic_position(self):
        gains, zero = self.found["none"]
        for key in ("6", "12", "24", "all"):
            self.assertLess(gains[key], 0, key)
            self.assertAlmostEqual(gains[key], zero[key], delta=0.15, msg=key)

    def test_a_stable_slope_gains_most_on_every_month_and_a_moving_one_on_short_windows(self):
        stable, moving = self.found["stable"][0], self.found["moving"][0]
        self.assertGreater(stable["all"], max(stable["6"], stable["12"]))
        self.assertGreater(stable["all"], 0.3)
        self.assertGreater(moving["6"], moving["all"])


def survey_rows(n=1600, seed=3):
    """alert_rows with a field of each kind the survey reads or names."""
    rows = alert_rows(n, seed)
    for k, r in enumerate(rows):
        r["state"].update(weekday=["Mon", "Tue", "Wed", "Thu", "Fri"][k % 5], side=["buy", "sell"][k % 2], rare="odd" if k % 200 == 0 else None,
                          const=1.5, mixed=1.0 if k % 2 else "one", obj={"a": 1}, empty=None)
    return rows


MANY = {"name": "many", "fields": ["x", "flag", "weekday", "time_utc"], "questions": [
    {"name": "skip", "type": "predicate", "rules": [{"text": "(a) Skip when x is above 1.", "expr": "x > 1"},
                                                   {"text": "(b) Skip when flag is set.", "expr": "flag"}]},
    {"name": "words", "type": "predicate", "rules": [{"text": "(c) Skip in the afternoon.", "expr": "time_utc >= '12:00'"},
                                                    "(d) Skip when it looks odd."]},
    {"name": "day", "type": "choice", "instructions": "Which day?", "choices": ["Mon", "Tue", "Wed"], "expr": "weekday"},
    {"name": "late", "type": "choice", "instructions": "Late in the week?", "choices": ["no", "yes"], "expr": "('no', 'yes')[weekday in ('Thu', 'Fri')]"},
    {"name": "size", "type": "score", "instructions": "How big is x?", "levels": ["small", "mid", "big"], "expr": "(x > -0.5) + (x > 0.5)"},
    {"name": "hunch", "type": "predicate", "instructions": "Does it feel risky?"}]}


class Command(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tt = TempTask(rows=survey_rows(), prompts={"many": MANY})
        cls.tt.ok("split")

    @classmethod
    def tearDownClass(cls):
        cls.tt.cleanup()

    def test_the_state_fields_give_factors_and_the_rest_is_named(self):
        r = self.tt.record("survey")
        names = {f["factor"] for f in r["by_class"][0]["factors"]}
        self.assertEqual(names, {"x", "noise", "maybe", "flag", "side = 'sell'", *(f"weekday = {d!r}" for d in ("Fri", "Mon", "Thu", "Tue", "Wed"))})
        self.assertEqual({s["factor"]: s["why"] for s in r["skipped"]},
                         {"time_utc": f"{len({x['state']['time_utc'] for x in self.tt.split_rows('train')})} distinct texts",
                          "series": "lists", "rare": "no level holds 2% to 98% of the units", "const": "constant",
                          "mixed": "mixed types", "obj": "objects", "empty": "no values"})
        maybe = next(f for f in r["by_class"][0]["factors"] if f["factor"] == "maybe")
        train = self.tt.split_rows("train")
        self.assertEqual(maybe["filled_pct"], round(100 * sum(x["state"]["maybe"] is not None for x in train) / len(train), 1))
        out = self.tt.ok("survey")
        for name in ("time_utc (", "series (lists)", "weekday = 'Mon'"):
            self.assertIn(name, out)

    def test_a_prompt_gives_its_rules_and_the_questions_the_machine_answers(self):
        r = self.tt.record("survey", self.tt.prompt("many"))
        factors = {f["factor"]: f["source"] for f in r["by_class"][0]["factors"]}
        self.assertEqual(factors, {"rule 1: (a) Skip when x is above 1.": "rule", "rule 2: (b) Skip when flag is set.": "rule",
                                   "skip: yes": "question", "rule 3: (c) Skip in the afternoon.": "rule",
                                   "day: Mon": "question", "day: Tue": "question", "day: Wed": "question",
                                   "late: yes": "question", "size: level": "question"})
        self.assertEqual({s["factor"]: s["why"][:7] for s in r["skipped"]},
                         {"rule 4: (d) Skip when it looks odd.": "no expr", "words": "a rule ", "hunch": "no expr"})
        self.assertEqual(r["factors_from"]["prompt"], "many")
        day = next(f for f in r["by_class"][0]["factors"] if f["factor"] == "day: Mon")
        self.assertAlmostEqual(day["filled_pct"], 60.0, delta=0.3)       # Thu and Fri are not options: refused, on no option

    def test_the_information_columns_take_the_sampling_noise_out_of_mu(self):
        """info% = 100 kappa mu2 / H(Z), mu2 = max(0, mu^2 - se^2), kappa = pi (1 - pi) / 2; nats/yr = units a year x kappa mu2."""
        r = self.tt.record("survey")
        block = r["by_class"][0]
        pi = block["units_in_class"] / block["units"]
        k, h_nats, per_year = pi * (1 - pi) / 2, -(pi * math.log(pi) + (1 - pi) * math.log(1 - pi)), 12 * block["units"] / r["months"]
        measured = [f for f in block["factors"] if f["mu"] is not None]
        for f in measured:
            mu2 = max(0.0, f["mu"] ** 2 - f["mu_se"] ** 2)
            self.assertAlmostEqual(f["info_pooled_pct"], 100 * k * mu2 / h_nats, delta=2e-3, msg=f["factor"])
            self.assertAlmostEqual(f["info_conditional_pct"], 100 * k * (mu2 + f["sigma"] ** 2) / h_nats, delta=2e-3, msg=f["factor"])
            self.assertAlmostEqual(f["nats_per_year"], per_year * k * mu2, delta=0.02, msg=f["factor"])
        self.assertTrue(any(abs(f["mu"]) < f["mu_se"] and f["info_pooled_pct"] == 0 for f in measured))

    def test_the_same_rows_give_the_same_record(self):
        self.assertEqual(self.tt.ok("survey", "--json"), self.tt.ok("survey", "--json"))

    def test_every_window_forecasts_the_same_months(self):
        r = self.tt.record("survey", "--gap", "3")
        block = r["by_class"][0]
        self.assertEqual(r["windows_months"], [6, 12, 24, None])
        self.assertEqual(block["predicted_months"]["first"], "2020-04")        # 2018-01 + 24 months + a gap of 3
        for f in block["factors"]:
            self.assertEqual({k: v["months"] for k, v in f["relearning"].items()}, dict.fromkeys(("6", "12", "24", "all"), block["predicted_months"]["count"]))
        self.assertEqual(self.tt.plab("survey", "--gap", "-1").code, 2)

    def test_the_brief_lists_the_survey(self):
        self.assertIn("plab survey", self.tt.ok("brief", "--agent", "a1"))


class Splits(unittest.TestCase):
    def setUp(self):
        self.tt = TempTask()
        self.tt.ok("split")

    def tearDown(self):
        self.tt.cleanup()

    def files(self):
        return {p.relative_to(self.tt.dir).as_posix(): p.read_bytes() for p in (self.tt.dir / ".plab").rglob("*") if p.is_file()}

    def test_the_training_survey_writes_nothing_and_reads_no_other_split(self):
        before = self.files()
        found = self.tt.ok("survey", "--json")
        self.assertEqual(self.files(), before)
        self.assertEqual(json.loads(found)["splits"], ["train"])
        for split in ("eval", "test"):
            (self.tt.dir / ".plab" / "data" / f"{split}.jsonl").unlink()
        self.assertEqual(self.tt.ok("survey", "--json"), found)
        for split in ("eval", "test"):
            (self.tt.dir / ".plab" / "data" / f"{split}.jsonl").write_text("this is not JSON\n{", encoding="utf-8")
        self.assertEqual(self.tt.ok("survey", "--json"), found)
        self.assertEqual(self.tt.ok("survey", self.tt.prompt("window"), "--json"), self.tt.ok("survey", self.tt.prompt("window"), "--json"))

    def test_a_short_span_keeps_the_windows_it_can_fill(self):
        self.tt.write_task(alert_task(splits={"train": ["2018-01-01", "2020-07-01"], "eval": ["2020-07-01", "2024-01-01"]}, half_boundary=None))
        self.tt.ok("split", "--force")
        r = self.tt.record("survey")
        self.assertEqual(r["windows_months"], [6, 12, None])                     # 24 is more than half of 30 months
        self.assertEqual(r["by_class"][0]["predicted_months"]["first"], "2019-01")


class ThreeClasses(unittest.TestCase):
    def test_each_class_is_surveyed_against_the_reference(self):
        tt = TempTask(rows=ticket_rows(), task=ticket_task(), prompts={})
        try:
            tt.ok("split")
            r = tt.record("survey")
            train = tt.split_rows("train")
            self.assertEqual([(b["class"], b["against"]) for b in r["by_class"]], [("billing", "other"), ("shipping", "other")])
            for block in r["by_class"]:
                self.assertEqual(block["units"], sum(x["label"] in ("other", block["class"]) for x in train))
                self.assertEqual(block["units_in_class"], sum(x["label"] == block["class"] for x in train))
                read = {f["factor"]: f for f in block["factors"]}
                mine = read[f"topic = {block['class']!r}"]              # the topic names the queue four times in five, in every year
                self.assertEqual((mine["reading"], mine["mu"] > 0, mine["relearning"]["all"]["gain_pct"] > 2 * mine["relearning"]["all"]["se_pct"]),
                                 ("stable", True, True))
                self.assertEqual(read["x"]["reading"], "none")
        finally:
            tt.cleanup()


if __name__ == "__main__":
    unittest.main()
