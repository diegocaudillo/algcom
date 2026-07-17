import unittest
from fractions import Fraction

from algcom import Algebra, BasisElement, SparseVector, TensorWord, Zero


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
        v = SparseVector({"y": 3})
        t = TensorWord( ("x","y") )
        self.assertTrue(isinstance(t, TensorWord))
        self.assertEqual(t.rank,2)

    def test_basis_element_helpers(self):
        left = BasisElement("x")
        right = BasisElement("x")
        self.assertEqual(left, right)
        self.assertEqual(hash(left), hash(right))
        self.assertEqual(str(left), "x")
        self.assertEqual(left.copy(), left)
        self.assertTrue(left < BasisElement("y"))

    def test_sparse_vector_accepts_numpy_arrays_as_basis_elements(self):
        import numpy as np

        matrix = np.array([[1, 0], [0, 1]], dtype=int)
        vector = SparseVector({matrix: 1})

        self.assertEqual(vector[matrix], Fraction(1, 1))
        self.assertEqual(vector[np.array([[1, 0], [0, 1]], dtype=int)], Fraction(1, 1))

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


if __name__ == "__main__":
    unittest.main()
