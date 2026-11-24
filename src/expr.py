"""The machine language of an `expr`: what the machine decider evaluates in place of a model.

The expression is parsed with `ast` and interpreted node by node; Python's `eval` is never used.
Allowed: field names, constants (numbers, strings, True/False/None), tuple/list literals,
comparisons (== != < <= > >= in, not in, chained), and/or/not, unary + and -,
the four arithmetic operators + - * / on numbers, the functions abs/min/max/len,
and subscripts (an index or a slice).

Missing data: when the expression reads a field (or a list element) whose value is None, or when
arithmetic or a comparison fails on the values (type mismatch, division by zero, max of an empty
list), it has no value. A rule without a value does not apply.
"""
from __future__ import annotations

import ast

MAX_CHARS = 2000
MAX_NODES = 400
FUNCS = {"abs": abs, "min": min, "max": max, "len": len}
_CMP = {
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}
_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED = (ast.Expression, ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List, ast.Compare, ast.BoolOp,
            ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Call, ast.Subscript,
            ast.Slice) + tuple(_CMP) + _BIN


class ExprError(ValueError):
    """The expression is not in the allowed subset."""


class _NoValue(Exception):
    """A value was None, or the computation failed on the data."""


class Expr:
    def __init__(self, src):
        if not isinstance(src, str) or not src.strip():
            raise ExprError("empty expression")
        if len(src) > MAX_CHARS:
            raise ExprError(f"expression longer than {MAX_CHARS} characters")
        try:
            tree = ast.parse(src.strip(), mode="eval")
        except SyntaxError as e:
            raise ExprError(f"syntax error: {e.msg}")
        n = 0
        names = set()
        func_nodes = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
        for node in ast.walk(tree):
            n += 1
            if not isinstance(node, _ALLOWED):
                raise ExprError(f"not allowed: {type(node).__name__}")
            if isinstance(node, ast.Constant) and not (node.value is None or isinstance(node.value, (bool, int, float, str))):
                raise ExprError(f"constant type not allowed: {type(node.value).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                    raise ExprError("only abs, min, max and len can be called")
                if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                    raise ExprError("function calls take plain positional arguments only")
                if not node.args:
                    raise ExprError(f"{node.func.id}() needs an argument")
            if isinstance(node, ast.Name) and id(node) not in func_nodes:
                names.add(node.id)
        if n > MAX_NODES:
            raise ExprError(f"expression too large ({n} nodes)")
        self.src = src.strip()
        self.tree = tree
        self.names = names

    def __repr__(self):
        return f"Expr({self.src!r})"

    def value(self, env):
        """The value over the fields in env, or None when it has none."""
        try:
            return _ev(self.tree.body, env)
        except _NoValue:
            return None

    def applies(self, env):
        """Whether a rule with this expression applies. Without a value it does not."""
        return bool(self.value(env))


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise _NoValue()
    return x


def _ev(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExprError(f"unknown name: {node.id}")
        v = env[node.id]
        if v is None:
            raise _NoValue()
        return v
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_ev(e, env) for e in node.elts)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for e in node.values:
                if not _ev(e, env):
                    return False
            return True
        for e in node.values:
            if _ev(e, env):
                return True
        return False
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not v
        v = _num(v) if not isinstance(v, bool) else int(v)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):
        a = _ev(node.left, env)
        b = _ev(node.right, env)
        a = int(a) if isinstance(a, bool) else _num(a)
        b = int(b) if isinstance(b, bool) else _num(b)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if b == 0:
            raise _NoValue()
        return a / b
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, env)
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(right, (tuple, list, str)):
                    raise _NoValue()
                if isinstance(right, str) and not isinstance(left, str):
                    raise _NoValue()
            try:
                ok = _CMP[type(op)](left, right)
            except TypeError:
                raise _NoValue()
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        f = FUNCS[node.func.id]
        args = [_ev(a, env) for a in node.args]
        if f is len:
            if len(args) != 1 or not isinstance(args[0], (tuple, list, str)):
                raise _NoValue()
            return len(args[0])
        if f is abs:
            if len(args) != 1:
                raise _NoValue()
            return abs(_num(args[0]) if not isinstance(args[0], bool) else int(args[0]))
        items = list(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
        if not items or any(x is None for x in items):
            raise _NoValue()
        try:
            return f(items)
        except TypeError:
            raise _NoValue()
    if isinstance(node, ast.Subscript):
        base = _ev(node.value, env)
        sl = node.slice
        if not isinstance(base, (tuple, list, str, dict)):
            raise _NoValue()
        if isinstance(sl, ast.Slice):
            if isinstance(base, dict):
                raise _NoValue()
            lo = _ev(sl.lower, env) if sl.lower is not None else None
            hi = _ev(sl.upper, env) if sl.upper is not None else None
            st = _ev(sl.step, env) if sl.step is not None else None
            for v in (lo, hi, st):
                if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
                    raise _NoValue()
            if st == 0:
                raise _NoValue()
            out = base[lo:hi:st]
            return tuple(out) if isinstance(out, list) else out
        idx = _ev(sl, env)
        try:
            v = base[idx]
        except (IndexError, KeyError, TypeError):
            raise _NoValue()
        if v is None:
            raise _NoValue()
        return tuple(v) if isinstance(v, list) else v
    raise ExprError(f"not allowed: {type(node).__name__}")
