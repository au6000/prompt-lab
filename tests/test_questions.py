"""The question kinds: how answers of each kind become what V reads, and how random answers are drawn."""
import math
import unittest

import numpy as np

from promptlab.questions import KINDS, Chose, Question, Refused, Scored, Space, Table, Yes

EPS = 1e-4
CLIPPED = math.log((1 - EPS) / EPS)


def question(kind, options):
    return Question("q", kind, "?", tuple(options), (None,) * len(options), (), None)


PREDICATE = question("predicate", ("no", "yes"))
CHOICE = question("choice", ("a", "b", "c", "d"))
SCORE = question("score", ("low", "mid", "high"))


def features(q, *answers):
    return Table.of([q], [(a,) for a in answers]).features()


class TestFeatures(unittest.TestCase):
    def test_a_predicate_is_read_by_its_logit_clipped_at_the_ends(self):
        got = features(PREDICATE, Yes(0.8), Yes(0.5), Yes(1.0), Yes(0.0))[:, 0]
        np.testing.assert_allclose(got, [math.log(0.8 / 0.2), 0.0, CLIPPED, -CLIPPED])

    def test_a_choice_is_read_by_each_option_against_the_first(self):
        got = features(CHOICE, Chose((0.1, 0.2, 0.3, 0.4), 3))
        np.testing.assert_allclose(got, [[math.log(2), math.log(3), math.log(4)]])

    def test_a_choice_named_without_probabilities_is_read_as_all_on_that_option(self):
        first, third = features(CHOICE, Chose(None, 0), Chose(None, 2))
        np.testing.assert_allclose(first, [-CLIPPED] * 3, atol=1e-3)
        np.testing.assert_allclose(third, [0, CLIPPED, 0], atol=1e-3)
        np.testing.assert_allclose(third, features(CHOICE, Chose((0.0, 0.0, 1.0, 0.0), 2))[0])

    def test_a_score_is_read_by_its_mean_level_as_one_column(self):
        got = features(SCORE, Scored((0.2, 0.5, 0.3), 1.1), Scored((1.0, 0.0, 0.0), 0.0))
        self.assertEqual(got.shape, (2, 1))
        np.testing.assert_allclose(got[:, 0], [math.log(0.55 / 0.45), -CLIPPED])

    def test_a_score_given_as_its_mean_alone_is_read_the_same(self):
        np.testing.assert_allclose(features(SCORE, Scored(None, 1.1)), features(SCORE, Scored((0.2, 0.5, 0.3), 1.1)))
        np.testing.assert_allclose(features(SCORE, Scored(None, 2.0)), [[CLIPPED]])

    def test_a_refused_answer_is_the_uniform_one_and_tells_v_nothing(self):
        for q in (PREDICATE, CHOICE, SCORE):
            table = Table.of([q], [(Refused(),)])
            np.testing.assert_allclose(table.rest[0], 1 / len(q.options))
            np.testing.assert_allclose(table.features(), 0, atol=1e-12)

    def test_the_table_marks_where_the_decider_refused(self):
        table = Table.of([PREDICATE, CHOICE], [(Yes(0.8), Refused()), (Refused(), Chose(None, 1))])
        self.assertEqual([m.tolist() for m in table.refused], [[False, True], [True, False]])
        self.assertEqual([m.tolist() for m in Table.random(table.spaces, np.random.default_rng(0), 3, sharp=False).refused],
                         [[False] * 3, [False] * 3])

    def test_what_an_answer_picks(self):
        rest = np.array([[0.7], [0.3]])
        self.assertEqual(KINDS["predicate"].picked(rest, 0.5).tolist(), [1, 0])
        self.assertEqual(KINDS["choice"].picked(np.array([[0.1, 0.2], [0.6, 0.3]]), 0.5).tolist(), [0, 1])
        self.assertEqual([KINDS["predicate"].chosen(Yes(1.0)), KINDS["choice"].chosen(Chose(None, 2)),
                          KINDS["score"].chosen(Scored(None, 2.0)), KINDS["choice"].chosen(Refused())], [1, 2, 2, -1])

    def test_the_questions_of_a_prompt_stand_side_by_side(self):
        table = Table.of([PREDICATE, CHOICE, SCORE], [(Yes(0.8), Chose(None, 1), Scored(None, 1.0))] * 5)
        self.assertEqual(table.features().shape, (5, 1 + 3 + 1))


class TestSharp(unittest.TestCase):
    def test_a_predicate_is_sharp_within_015_of_either_end_as_in_010(self):
        p = np.array([0.0, 0.1, 0.15, 0.15000000000000002, 0.5, 0.85, 0.8499999999999999, 0.86, 1.0, 1 - 0.15, 0.3 / 2])
        table = Table((Space("predicate", 2),), (p[:, None],))
        np.testing.assert_array_equal(table.sharp(), np.minimum(p, 1 - p) <= 0.15)

    def test_a_choice_is_sharp_when_one_option_has_nearly_all_the_probability(self):
        rest = np.array([[0.9, 0.05, 0.0], [0.03, 0.03, 0.04], [0.5, 0.3, 0.1], [0.0, 0.0, 1.0]])
        table = Table((Space("choice", 4),), (rest,))
        np.testing.assert_array_equal(table.sharp(), [True, True, False, True])

    def test_every_question_must_be_sharp(self):
        sharp, soft = np.array([[1.0], [0.0]]), np.array([[0.5], [0.0]])
        table = Table((Space("predicate", 2), Space("predicate", 2)), (sharp, soft))
        np.testing.assert_array_equal(table.sharp(), [False, True])


class TestRandomAnswers(unittest.TestCase):
    def test_one_predicate_draws_what_010_drew(self):
        """soft = random(n); then a share in [0.05, 0.6] and sharp = random(n) < share."""
        n, spaces = 50, (Space("predicate", 2),)
        rng, old = np.random.default_rng(777), np.random.default_rng(777)
        soft = Table.random(spaces, rng, n, sharp=False)
        sharp = Table.random(spaces, rng, n, sharp=True)
        np.testing.assert_array_equal(soft.rest[0][:, 0], old.random(n))
        share = old.uniform(0.05, 0.6)
        np.testing.assert_array_equal(sharp.rest[0][:, 0], (old.random(n) < share).astype(float))

    def test_random_answers_have_the_shape_of_the_questions(self):
        spaces = (Space("choice", 4), Space("score", 3), Space("predicate", 2))
        rng = np.random.default_rng(1)
        for sharp in (False, True):
            table = Table.random(spaces, rng, 200, sharp=sharp)
            self.assertEqual([r.shape for r in table.rest], [(200, 3), (200, 2), (200, 1)])
            self.assertEqual(table.features().shape, (200, 3 + 1 + 1))
            self.assertEqual(bool(table.sharp().all()), sharp)
            for r in table.rest:
                self.assertTrue((r >= 0).all() and (r.sum(axis=1) <= 1 + 1e-12).all())


if __name__ == "__main__":
    unittest.main()
