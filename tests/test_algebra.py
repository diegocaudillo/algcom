import unittest

from algcom.algebra import Algebra
from algcom.core import SparseVector


class AlgebraTests(unittest.TestCase):
    def test_algebra_from_rule_and_fold_operations(self):
        algebra = Algebra.from_rule(
            rule=lambda n, m: [n + m],
            one=SparseVector({0: 1}),
            degree_method=lambda n: n,
            max_degree=5,
        )

        left = SparseVector({1: 1})
        right = SparseVector({2: 1})

        self.assertEqual(algebra.multiply(left, right), SparseVector({3: 1}))
        self.assertEqual(algebra.fold_left([left, right]), SparseVector({3: 1}))
        self.assertEqual(algebra.fold_right([left, right]), SparseVector({3: 1}))

    def test_algebra_series_helpers_on_scalar_unit(self):
        algebra = Algebra.from_rule(
            rule=lambda n, m: [n + m],
            one=SparseVector({0: 1}),
            degree_method=lambda n: n,
            max_degree=3,
        )

        one = algebra.unit(1)
        self.assertEqual(str(algebra.exponential(one)), "⟅8 0⟆/3")
        self.assertEqual(str(algebra.logarithm(one)), "0")
        self.assertEqual(str(algebra.geometric(one)), "⟅4 0⟆")


if __name__ == "__main__":
    unittest.main()
