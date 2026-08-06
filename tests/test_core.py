import unittest
from fractions import Fraction

from algcom.core import BasisElement, SparseVector, Zero, LinearLift


class CoreTests(unittest.TestCase):
    def test_basis_element_round_trip_and_equality(self):
        left = BasisElement("x")
        right = BasisElement("x")
        self.assertEqual(left, right)
        self.assertEqual(left, "x")
        self.assertEqual(str(left), "x")
        self.assertEqual(repr(left), "BasisElement('x')")

    def test_sparse_vector_add_sub_mul_and_string_form(self):
        left = SparseVector({"x": 2, "y": -1})
        right = SparseVector({"x": 1, "z": 3})

        self.assertEqual(left + right, SparseVector({"x": 3, "y": -1, "z": 3}))
        self.assertEqual(left - right, SparseVector({"x": 1, "y": -1, "z": -3}))
        self.assertEqual(left * 2, SparseVector({"x": 4, "y": -2}))
        self.assertEqual(2 * left, SparseVector({"x": 4, "y": -2}))
        self.assertEqual(str(left), "⟅2 x - y⟆")

    def test_sparse_vector_zero_and_integer_coefficients(self):
        vector = SparseVector({"x": Fraction(1, 2), "y": Fraction(2, 3)})
        d, scaled = vector.integer_coefficients()

        self.assertEqual(d, 6)
        self.assertEqual(scaled, SparseVector({"x": 3, "y": 4}))
        self.assertEqual(SparseVector() + Zero, SparseVector())
        self.assertEqual((SparseVector() - SparseVector())._data, {})

    def test_sparse_vector_dual_bracket_projection_and_colinearity(self):
        left = SparseVector({"x": 2, "y": -1})
        right = SparseVector({"x": 1, "y": 1})

        self.assertEqual(left.bracket(right), Fraction(1))
        self.assertEqual(left.dual()(right), Fraction(1))
        self.assertEqual(left.projection(right), Fraction(1, 5))
        self.assertTrue(left.iscolinear(SparseVector({"x": 4, "y": -2})))
        self.assertFalse(left.iscolinear(SparseVector({"x": 1, "y": 1})))

    def test_linear_lift_applies_a_function_to_each_basis_key(self):
        lifted = LinearLift(lambda value: {"x": value})
        vector = SparseVector({"a": 1})

        with self.assertRaises(ValueError):
            lifted(vector)


if __name__ == "__main__":
    unittest.main()
