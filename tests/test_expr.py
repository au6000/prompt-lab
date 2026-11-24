import time
import unittest

from promptlab.expr import Expr, ExprError


class TestAllowed(unittest.TestCase):
    def v(self, src, **env):
        return Expr(src).value(env)

    def a(self, src, **env):
        return Expr(src).applies(env)

    def test_arithmetic_and_comparisons(self):
        self.assertEqual(self.v("a + b * 2 - c / 4", a=1, b=2, c=8), 3.0)
        self.assertTrue(self.a("1 < a <= 3", a=3))
        self.assertFalse(self.a("1 < a <= 3", a=4))
        self.assertTrue(self.a("-a > -5", a=4))
        self.assertTrue(self.a("not flag", flag=False))
        self.assertTrue(self.a("a > 1 and (b or c)", a=2, b=False, c=True))
        self.assertTrue(self.a("x != 'Mon'", x="Tue"))

    def test_strings_membership_functions_subscripts(self):
        self.assertTrue(self.a("'09:00' <= t <= '10:29'", t="09:45"))
        self.assertFalse(self.a("'09:00' <= t <= '10:29'", t="10:30"))
        self.assertTrue(self.a("d in ('Mon', 'Fri')", d="Fri"))
        self.assertTrue(self.a("d not in ['Mon']", d="Fri"))
        self.assertEqual(self.v("abs(a)", a=-3), 3)
        self.assertEqual(self.v("max(s)", s=[1, 5, 2]), 5)
        self.assertEqual(self.v("min(a, b)", a=4, b=2), 2)
        self.assertEqual(self.v("len(s)", s=[1, 2, 3]), 3)
        self.assertEqual(self.v("s[-1]", s=[1, 2, 3]), 3)
        self.assertEqual(self.v("max(s[-3:])", s=[9, 1, 2, 3]), 3)
        self.assertEqual(self.v("d['k']", d={"k": 7}), 7)

    def test_names_exclude_functions(self):
        self.assertEqual(Expr("max(s) > a and len(s) > 2").names, {"s", "a"})


class TestMissingData(unittest.TestCase):
    def test_none_means_the_rule_does_not_apply(self):
        e = Expr("a > 1")
        self.assertFalse(e.applies({"a": None}))
        self.assertIsNone(e.value({"a": None}))
        self.assertFalse(Expr("not a").applies({"a": None}))          # even a negation does not apply
        self.assertFalse(Expr("a == None").applies({"a": None}))
        self.assertFalse(Expr("max(s) > 0").applies({"s": [1, None, 2]}))
        self.assertFalse(Expr("s[1] > 0").applies({"s": [1, None]}))

    def test_failures_mean_the_rule_does_not_apply(self):
        self.assertFalse(Expr("a / b > 1").applies({"a": 1, "b": 0}))
        self.assertFalse(Expr("a > 1").applies({"a": "text"}))
        self.assertFalse(Expr("max(s) > 1").applies({"s": []}))
        self.assertFalse(Expr("s[5] > 1").applies({"s": [1]}))


class TestRefused(unittest.TestCase):
    BAD = ["a.__class__", "open('x')", "__import__('os')", "(lambda: 1)()", "[x for x in s]", "a ** 2", "a // 2",
           "a % 2", "max(s, key=abs)", "max(*s)", "f'{a}'", "{'a': 1}", "1 if a else 2", "(y := 3)", "b'x'",
           "exec('1')", "getattr(a, 'x')", "a; b", "import os", "s[::]", ""]

    def test_disallowed_syntax(self):
        for src in self.BAD:
            if src == "s[::]":
                continue
            with self.assertRaises(ExprError, msg=src):
                Expr(src)

    def test_slices_without_bounds_are_fine(self):
        self.assertEqual(Expr("len(s[::])").value({"s": [1, 2]}), 2)

    def test_no_string_multiplication_blowup(self):
        t = time.time()
        self.assertFalse(Expr("'a' * 100000000 == 'b'").applies({}))
        self.assertLess(time.time() - t, 1.0)

    def test_size_limits(self):
        with self.assertRaises(ExprError):
            Expr("a + " * 1000 + "a")

    def test_unknown_name_is_an_error_not_a_miss(self):
        with self.assertRaises(ExprError):
            Expr("zz > 1").value({"a": 1})


if __name__ == "__main__":
    unittest.main()
