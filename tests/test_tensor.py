import unittest

from fractions import Fraction

from algcom.core import SparseVector
from algcom.tensor import SparseTensor, TensorWord


class TensorTests(unittest.TestCase):
    def test_tensor_word_flattens_nested_words_and_tracks_rank(self):
        left = TensorWord(("x", "y"))
        right = TensorWord((left, "z"))

        self.assertEqual(right.rank, 3)
        self.assertEqual(right.value, ("x", "y", "z"))

    def test_sparse_tensor_builds_from_vectors_and_multilinear_products(self):
        left = SparseTensor(["x", "y"])
        right = SparseTensor(["u", "v"])

        self.assertEqual(left.min_rank(), 1)
        self.assertEqual(left.max_rank(), 1)

        product = SparseTensor.multilinear([left, right])
        self.assertEqual(product.max_rank(), 2)
        self.assertEqual(product.min_rank(), 2)
        self.assertEqual(product[TensorWord(("x", "u"))], 1)
        self.assertEqual(product[TensorWord(("y", "v"))], 1)

    def test_tensor_word_functions_evaluate_entrywise(self):
        word = TensorWord((lambda x: x + 1, lambda x: x * 2))
        point = TensorWord((2, 3))

        self.assertEqual(word.evaluate_at(point).value, (3, 6))
        self.assertEqual(word.pre_compose(TensorWord((lambda x: x + 1, lambda x: x * 2))).value, (4, 8))

    def test_sparse_tensor_unit_and_bilinear_construction(self):
        unit = SparseTensor.unit(Fraction(2, 3))
        self.assertEqual(unit[TensorWord()], Fraction(2, 3))

        left = SparseVector({"x": 2})
        right = SparseVector({"y": 3})
        bilinear = SparseTensor.bilinear(left, right)

        self.assertEqual(bilinear[TensorWord(("x", "y"))], Fraction(6))
        self.assertEqual(bilinear.min_rank(), 2)
        self.assertEqual(bilinear.max_rank(), 2)
