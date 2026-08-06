import unittest

from algcom.series import ExponentialSeries, ExponentialTerm, OrdinarySeries, OrdinaryTerm
from algcom.core import SparseVector


class SeriesTests(unittest.TestCase):
    def test_ordinary_series_multiplication_matches_convolution(self):
        left = SparseVector(OrdinaryTerm(1))
        right = SparseVector(OrdinaryTerm(2))

        self.assertEqual(OrdinarySeries.multiply(left, right), SparseVector(OrdinaryTerm(3)))

    def test_exponential_series_multiplication_matches_binomial_rule(self):
        left = SparseVector(ExponentialTerm(1))
        right = SparseVector(ExponentialTerm(2))

        self.assertEqual(ExponentialSeries.multiply(left, right), SparseVector({ExponentialTerm(3): 3}))


if __name__ == "__main__":
    unittest.main()
