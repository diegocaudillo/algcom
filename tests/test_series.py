import unittest
from fractions import Fraction
from math import factorial

from algcom.series import OrdinarySeries, ExponentialSeries


def _normal_moment(n: int, variance: int = 1) -> Fraction:
    if n % 2 == 1:
        return Fraction(0)
    return Fraction(factorial(n), 2 ** (n // 2) * factorial(n // 2)) * Fraction(variance) ** (n // 2)


class OrdinarySeriesTests(unittest.TestCase):
    def test_plain_convolution(self):
        p = OrdinarySeries({0: 1, 1: 1})
        q = OrdinarySeries({0: 1, 1: -1})
        self.assertEqual(p.multiply(q), OrdinarySeries({0: 1, 2: -1}))


class ExponentialSeriesTests(unittest.TestCase):
    def test_binomial_convolution_on_basis_elements(self):
        # t^1 . t^1 = C(2,1) t^2 = 2 t^2
        self.assertEqual(
            ExponentialSeries({1: 1}).multiply(ExponentialSeries({1: 1})),
            ExponentialSeries({2: 2}),
        )
        # t^1 . t^2 = C(3,1) t^3 = 3 t^3
        self.assertEqual(
            ExponentialSeries({1: 1}).multiply(ExponentialSeries({2: 1})),
            ExponentialSeries({3: 3}),
        )

    def test_normal_distribution_cumulants(self):
        order = 8
        moments = {n: _normal_moment(n) for n in range(order + 1)}
        M = ExponentialSeries.from_moments(moments)
        K = M.cumulants(degree=order)
        expected = ExponentialSeries({2: 1})
        self.assertEqual(K, expected)

    def test_moment_cumulant_round_trip(self):
        order = 8
        moments = {n: _normal_moment(n) for n in range(order + 1)}
        M = ExponentialSeries.from_moments(moments)
        M_back = ExponentialSeries.from_cumulants({2: 1}, degree=order)
        self.assertEqual(M_back, M.truncate(order))

    def test_thiele_additivity_of_cumulants(self):
        order = 6
        M1 = ExponentialSeries.from_moments({n: _normal_moment(n) for n in range(order + 1)})
        M2 = ExponentialSeries.from_moments({n: _normal_moment(n) for n in range(order + 1)})
        Msum = M1.multiply(M2).truncate(order)
        expected_sum = ExponentialSeries.from_moments({n: _normal_moment(n, variance=2) for n in range(order + 1)})
        self.assertEqual(Msum, expected_sum)

        K1 = M1.cumulants(degree=order)
        K2 = M2.cumulants(degree=order)
        Ksum = Msum.cumulants(degree=order)
        self.assertEqual(K1 + K2, Ksum)


if __name__ == "__main__":
    unittest.main()
