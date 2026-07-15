import unittest
from fractions import Fraction

from algcom import Algebra, BasisElement, SparseVector, Tensor, Zero, tensor


class SparseVectorTests(unittest.TestCase):
    def test_sparse_vector_defaults_to_zero_and_accumulates(self):
        v = SparseVector()
        v["x"] += 2
        v["x"] += 3
        self.assertEqual(v["x"], 5)
        self.assertEqual(v["y"], 0)

    def test_sparse_vector_arithmetic_and_zero_pruning(self):
        u = SparseVector({"x": Fraction(1, 2), "y": 1})
        v = SparseVector({"x": Fraction(1, 2), "z": 2})
        self.assertEqual(u + v, SparseVector({"x": 1, "y": 1, "z": 2}))
        self.assertEqual(u - v, SparseVector({"y": 1, "z": -2}))
        self.assertEqual(2 * u, SparseVector({"x": 1, "y": 2}))
        self.assertEqual(u * 3, SparseVector({"x": Fraction(3, 2), "y": 3}))
        self.assertEqual(u + Zero, u)

    def test_string_and_equality_ignore_order(self):
        left = SparseVector({"x": 2, "y": -1})
        right = SparseVector({"y": -1, "x": 2})
        self.assertEqual(left, right)
        self.assertIn("x", str(left))

    def test_tensor_from_elements_and_vectors(self):
        u = SparseVector({"x": 2})
        t = tensor(u)
        self.assertTrue(isinstance(t, Tensor))
        expected = Tensor()
        expected[("x",)] = 2
        self.assertEqual(t, expected)

        v = SparseVector({"y": 3})
        tv = tensor(u, v)
        expected = Tensor()
        expected[("x", "y")] = 6
        self.assertEqual(tv, expected)

    def test_tensor_concatenation_and_rank(self):
        left = tensor(SparseVector({"x": 1}), SparseVector({"y": 1}))
        right = tensor(SparseVector({"z": 1}))
        combined = tensor(left, right)
        expected = Tensor()
        expected[("x", "y", "z")] = 1
        self.assertEqual(combined, expected)
        self.assertTrue(combined.is_pure_of_rank(3))

    def test_tensor_unit_and_duality(self):
        unit = Tensor.unit()
        expected_unit = Tensor()
        expected_unit[()] = 1
        self.assertEqual(unit, expected_unit)
        self.assertTrue(unit.is_pure_of_rank(0))

        left = SparseVector({"x": Fraction(1, 2), "y": 1})
        right = SparseVector({"x": 2, "z": 3})
        self.assertEqual(left.bracket(right), Fraction(1))
        self.assertEqual(left.dual()(right), Fraction(1))

    def test_basis_element_helpers(self):
        left = BasisElement("x")
        right = BasisElement("x")
        self.assertEqual(left, right)
        self.assertEqual(hash(left), hash(right))
        self.assertEqual(str(left), "x")
        self.assertEqual(left.copy(), left)
        self.assertTrue(left < BasisElement("y"))

    def test_algebra_scaffold(self):
        def product_rule(left, right):
            return SparseVector({f"{left}{right}": 1})

        algebra = Algebra(product=product_rule)
        left = SparseVector({"x": 1})
        middle = SparseVector({"y": 1})
        right = SparseVector({"z": 1})

        self.assertEqual(algebra.multiply(left, middle), SparseVector({"xy": 1}))
        self.assertEqual(algebra.left_fold([left, middle, right]), SparseVector({"xyz": 1}))
        self.assertEqual(algebra.right_fold([left, middle, right]), SparseVector({"xyz": 1}))

    def test_algebra_operator_helpers(self):
        algebra = Algebra(product=lambda left, right: SparseVector({f"{left}{right}": 1}))
        left = SparseVector({"x": 1})
        right = SparseVector({"y": 1})

        self.assertEqual(algebra.twist(left, right), SparseVector({"yx": 1}))
        self.assertEqual(algebra.commutator(left, right), SparseVector({"xy": 1}) - SparseVector({"yx": 1}))
        self.assertEqual(algebra.lie_bracket(left, right), SparseVector({"xy": 1}) - SparseVector({"yx": 1}))

    def test_series_and_operator_checks(self):
        algebra = Algebra()
        algebra.set_product(lambda left, right: SparseVector({f"{left}{right}": 1}))
        element = SparseVector({"x": 1})

        expected = SparseVector()
        expected[()] = 1
        expected["()x"] = 1
        expected["()xx"] = Fraction(1, 2)
        self.assertEqual(algebra.exponential(element, order=2), expected)

        expected = SparseVector()
        expected[()] = 1
        expected["()x"] = 1
        expected["()xx"] = 1
        self.assertEqual(algebra.polynomial(element, order=2), expected)

        expected = SparseVector()
        expected["()x"] = 1
        expected["()xx"] = -Fraction(1, 2)
        self.assertEqual(algebra.logarithm(element, order=2), expected)

        expected = SparseVector()
        expected[()] = 1
        expected["()x"] = 1
        expected["()xx"] = 1
        self.assertEqual(algebra.geometric(element, order=2), expected)

        operator = lambda value: value * 0
        self.assertTrue(algebra.is_multiplicative(operator, [element, element]))
        self.assertTrue(algebra.is_rota_leibniz(operator, [element, element]))
        self.assertTrue(algebra.is_rota_baxter(operator, [element, element]))


if __name__ == "__main__":
    unittest.main()
