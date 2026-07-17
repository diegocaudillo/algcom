import io
import unittest
from contextlib import redirect_stdout
from fractions import Fraction

from algcom.polynomial import Polynomial, _poly_little_test


class PolynomialTests(unittest.TestCase):
    def test_polynomial_factorization_and_evaluation(self):
        output = io.StringIO()
        with redirect_stdout(output):
            _poly_little_test()

        text = output.getvalue()
        self.assertIn("Factorisation:", text)
        self.assertIn("Evaluation p(-1)=", text)
        self.assertIn("Evaluation p( 0)=", text)
        self.assertIn("Evaluation p( 1)=", text)
        self.assertIn("log", text)
        self.assertNotIn("[FAIL]", text)

    def test_polynomial_multiplication_and_evaluation(self):
        px = Polynomial({0: -1, 1: 0, 2: 1})
        q1 = Polynomial({0: 1, 1: 1})
        q2 = Polynomial({0: -1, 1: 1})

        q = q1.multiply(q2)
        self.assertEqual(q, px)
        self.assertEqual(px.evaluate_at(-1), 0)
        self.assertEqual(px.evaluate_at(0), -1)
        self.assertEqual(px.evaluate_at(1), 0)

    def test_polynomial_logarithm_and_exponential(self):
        one = Polynomial.one()
        x = Polynomial({1: 1})

        logarithm = Polynomial.algebra().logarithm(one + x, order=5)
        expected_log = Polynomial({
            1: Fraction(1, 1),
            2: Fraction(-1, 2),
            3: Fraction(1, 3),
            4: Fraction(-1, 4),
            5: Fraction(1, 5),
        })
        self.assertEqual(logarithm, expected_log)

        exponential = Polynomial.algebra().exponential(x, order=5)
        expected_exp = Polynomial({
            0: Fraction(1, 1),
            1: Fraction(1, 1),
            2: Fraction(1, 2),
            3: Fraction(1, 6),
            4: Fraction(1, 24),
            5: Fraction(1, 120),
        })
        self.assertEqual(exponential, expected_exp)
