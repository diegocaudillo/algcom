from __future__ import annotations

from typing import Callable, Optional, Sequence

from .core import SparseVector
from .tensor import Tensor, tensor


class Algebra:
    """A minimalist algebra scaffold that can hold a product and fold it."""

    def __init__(self, product: Optional[Callable[[SparseVector, SparseVector], SparseVector]] = None):
        self.product = product or (lambda left, right: tensor(left, right))

    def _bilinear_lift(self, rule: Callable[[object, object], object]) -> Callable[[SparseVector, SparseVector], SparseVector]:
        def lifted(left: SparseVector, right: SparseVector) -> SparseVector:
            result = SparseVector()
            for left_key, left_coeff in left._data.items():
                for right_key, right_coeff in right._data.items():
                    outcome = rule(left_key, right_key)
                    if isinstance(outcome, SparseVector):
                        for out_key, out_coeff in outcome._data.items():
                            result[out_key] += left_coeff * right_coeff * out_coeff
                    elif isinstance(outcome, (list, tuple)):
                        for item in outcome:
                            result[item] += left_coeff * right_coeff
                    else:
                        result[outcome] += left_coeff * right_coeff
            return result

        return lifted

    def set_product(self, rule: Callable[[object, object], object]) -> None:
        self.product = self._bilinear_lift(rule)

    def multiply(self, left: SparseVector, right: SparseVector) -> SparseVector:
        return self.product(left, right)

    def left_fold(self, elements: Sequence[SparseVector]) -> SparseVector:
        if not elements:
            return Tensor.unit()
        result = elements[0]
        for element in elements[1:]:
            result = self.multiply(result, element)
        return result

    def right_fold(self, elements: Sequence[SparseVector]) -> SparseVector:
        if not elements:
            return Tensor.unit()
        result = elements[-1]
        for element in reversed(elements[:-1]):
            result = self.multiply(element, result)
        return result
