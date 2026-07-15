from __future__ import annotations

from typing import Callable, Optional, Sequence

from fractions import Fraction

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

    def twist(self, left: SparseVector, right: SparseVector) -> SparseVector:
        return self.multiply(right, left)

    def commutator(self, left: SparseVector, right: SparseVector) -> SparseVector:
        return self.multiply(left, right) - self.multiply(right, left)

    def lie_bracket(self, left: SparseVector, right: SparseVector) -> SparseVector:
        return self.commutator(left, right)

    def _power_series(self, element: SparseVector, order: int = 3) -> list[SparseVector]:
        series = []
        current = Tensor.unit().as_sparse_vector()
        series.append(current)
        for _ in range(1, order + 1):
            current = self.multiply(current, element)
            series.append(current)
        return series

    def exponential(self, element: SparseVector, order: int = 3) -> SparseVector:
        series = self._power_series(element, order)
        result = SparseVector()
        for index, term in enumerate(series):
            if index == 0:
                result += Tensor.unit().as_sparse_vector()
            else:
                result += term * Fraction(1, index)
        return result

    def polynomial(self, element: SparseVector, order: int = 3) -> SparseVector:
        series = self._power_series(element, order)
        result = SparseVector()
        for index, term in enumerate(series):
            result += term * Fraction(1, 1)
        return result

    def logarithm(self, element: SparseVector, order: int = 3) -> SparseVector:
        series = self._power_series(element, order)
        result = SparseVector()
        for index, term in enumerate(series[1:], start=1):
            if index > 0:
                result += term * Fraction((-1) ** (index + 1), index)
        return result

    def geometric(self, element: SparseVector, order: int = 3) -> SparseVector:
        series = self._power_series(element, order)
        result = SparseVector()
        for index, term in enumerate(series):
            result += term * Fraction(1, 1)
        return result

    def is_multiplicative(self, operator: Callable[[SparseVector], SparseVector], sample: Sequence[SparseVector]) -> bool:
        if len(sample) < 2:
            return True
        left, right = sample[0], sample[1]
        return operator(self.multiply(left, right)) == self.multiply(operator(left), operator(right))

    def is_rota_leibniz(self, operator: Callable[[SparseVector], SparseVector], sample: Sequence[SparseVector]) -> bool:
        if len(sample) < 2:
            return True
        left, right = sample[0], sample[1]
        return operator(self.multiply(left, right)) == self.multiply(operator(left), right) + self.multiply(left, operator(right))

    def is_rota_baxter(self, operator: Callable[[SparseVector], SparseVector], sample: Sequence[SparseVector]) -> bool:
        if len(sample) < 2:
            return True
        left, right = sample[0], sample[1]
        return self.multiply(operator(left), operator(right)) == operator(self.multiply(operator(left), right) + self.multiply(left, operator(right)))
