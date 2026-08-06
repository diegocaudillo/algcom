import unittest

from algcom.core import SparseVector
from algcom.tensor import SparseTensor, TensorWord


class TensorTests(unittest.TestCase):
    def test_tensor_word_flattens_and_formats(self):
        word = TensorWord(("a", TensorWord(("b", "c"))))

        self.assertEqual(word.rank, 3)
        self.assertEqual(str(word), "a⊗b⊗c")

    def test_sparse_tensor_construction_and_bilinear_multilinear_ops(self):
        tensor = SparseTensor([("a", "b"), ("c",)])
        self.assertEqual(tensor.max_rank(), 2)
        self.assertEqual(tensor.min_rank(), 1)

        left = SparseVector({"a": 1})
        right = SparseVector({"b": 2})
        bilinear = SparseTensor.bilinear(left, right)
        multilinear = SparseTensor.multilinear([
            SparseVector({"a": 1}),
            SparseVector({"b": 2}),
            SparseVector({"c": 3}),
        ])

        self.assertEqual(str(bilinear), "⟅2 a⊗b⟆")
        self.assertEqual(str(multilinear), "⟅6 a⊗b⊗c⟆")
        self.assertEqual(bilinear.max_rank(), 2)
        self.assertEqual(multilinear.max_rank(), 3)


if __name__ == "__main__":
    unittest.main()
