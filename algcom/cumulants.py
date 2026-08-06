from __future__ import annotations

from math import factorial
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .algebra import Algebra
from .core import LinearLift, SparseVector


class _PartitionScheme:
    """Generate set partitions and associated Möbius-type coefficients."""

    def __init__(self, kind: str = "classical"):
        self.kind = (kind or "classical").lower()

    def partitions(self, n: int) -> List[Tuple[Tuple[int, ...], ...]]:
        if n <= 0:
            return [tuple()]

        elements = tuple(range(n))

        def build(remaining: Tuple[int, ...], current: List[List[int]]):
            if not remaining:
                normalized = []
                for block in current:
                    normalized.append(tuple(sorted(block)))
                yield tuple(sorted(normalized, key=lambda block: block))
                return

            first = remaining[0]
            for index in range(len(current)):
                updated = [list(block) for block in current]
                updated[index].append(first)
                yield from build(remaining[1:], updated)

            updated = [list(block) for block in current] + [[first]]
            yield from build(remaining[1:], updated)

        return list(build(elements[1:], [[elements[0]]]))

    def moebius_coefficient(self, partition: Sequence[Sequence[int]]) -> int | float:
        size = len(partition)
        if self.kind == "classical":
            return (-1) ** (size - 1) * factorial(size - 1) if size > 0 else 1
        if self.kind == "boolean":
            return (-1) ** (size - 1) if size > 0 else 1
        if self.kind == "monotone":
            return factorial(size - 1) if size > 0 else 1
        if self.kind == "anti-monotone":
            return (-1) ** (size - 1) if size > 0 else 1
        if self.kind == "free":
            if size <= 1:
                return 1
            total = 1
            for block in partition:
                length = len(block)
                total *= (-1) ** (length - 1) * self._catalan(length - 1)
            return total
        raise ValueError(f"unsupported cumulant type: {self.kind}")

    @staticmethod
    def _catalan(n: int) -> int:
        if n <= 0:
            return 1
        result = 0
        for k in range(n):
            result += _PartitionScheme._catalan(k) * _PartitionScheme._catalan(n - 1 - k)
        return result


class Cumulants:
    """Combinatorial cumulants between two sparse-vector algebras."""

    def __init__(self, algebra_a: Algebra, algebra_b: Algebra, umbra: Callable[[SparseVector], SparseVector]):
        self.algebra_a = algebra_a
        self.algebra_b = algebra_b
        self.umbra = self._prepare_umbra(umbra)

    @staticmethod
    def _prepare_umbra(umbra: Callable[[SparseVector], SparseVector]) -> Callable[[SparseVector], SparseVector]:
        if not callable(umbra):
            raise TypeError("umbra must be callable")

        def wrapped(value: SparseVector) -> SparseVector:
            if isinstance(value, SparseVector):
                try:
                    return umbra(value)
                except (TypeError, AttributeError):
                    return LinearLift(umbra)(value)
            return SparseVector(umbra(value))

        return wrapped

    def _fold(self, elements: Iterable[SparseVector], fold: str = "left") -> SparseVector:
        if fold == "right":
            return self.algebra_a.fold_right(list(elements)) 
        if fold == "left":
            return self.algebra_a.fold_left(list(elements)) 
        raise ValueError("fold must be 'left' or 'right'")

    def _product(self, elements: Sequence[SparseVector], algebra: Algebra, fold: str = "left") -> SparseVector:
        if not elements:
            return algebra.unit(1)
        if fold == "right":
            return algebra.fold_right(list(elements))
        return algebra.fold_left(list(elements))

    def _rank_of(self, item: object) -> Optional[int]:
        value = getattr(item, "value", item)
        if self.algebra_b.degree_method is None:
            return None
        try:
            return self.algebra_b.degree_method(value)
        except TypeError:
            return None

    def kappa(self, block: Sequence[SparseVector]) -> SparseVector:
        '''
        Replace by hand-crafted combinatorial cumulant for a block of variables.
        eg. 
        C = Cumulants(A,B,umbra)
        C.kappa = lambda block : CombinatorialCumulant(block) 
        '''
        return self.moment_cumulant(block, kind="classical", fold="left")

    def moment_cumulant(
        self,
        variables: Sequence[SparseVector],
        order: Optional[int] = None,
        kind: str = "classical",
        fold: str = "left",
    ) -> SparseVector:
        """Compute a moment-to-cumulant transform over a list of sparse vectors."""
        if order is None:
            order = len(variables)
        if order < 0:
            raise ValueError("order must be non-negative")

        items = list(variables)[:order]
        if not items:
            return self.algebra_b.unit(0)

        scheme = _PartitionScheme(kind)
        result = self.algebra_b.unit(0)
        for partition in scheme.partitions(len(items)):
            coefficient = scheme.moebius_coefficient(partition)
            block_terms = []
            for block in partition:
                block_values = [items[index] for index in block]
                product_in_a = self._product(block_values, self.algebra_a, fold=fold)
                block_terms.append(self.umbra(product_in_a))
            combined = self._product(block_terms, self.algebra_b, fold=fold)
            result += coefficient * combined
        return result

    def cumulant_moment(
        self,
        variables: Sequence[SparseVector],
        order: Optional[int] = None,
        kind: str = "classical",
        fold: str = "left",
    ) -> SparseVector:
        """Compute a cumulant-to-moment transform over a list of sparse vectors."""
        if order is None:
            order = len(variables)
        if order < 0:
            raise ValueError("order must be non-negative")

        items = list(variables)[:order]
        if not items:
            return self.algebra_b.unit(1)

        scheme = _PartitionScheme(kind)
        result = self.algebra_b.unit(0)
        for partition in scheme.partitions(len(items)):
            block_terms = []
            for block in partition:
                block_values = [items[index] for index in block]
                block_terms.append(self.kappa(block_values))
            combined = self._product(block_terms, self.algebra_b, fold=fold)
            result += combined
        return result

    def cumulants(
        self,
        variables: Sequence[SparseVector],
        order: Optional[int] = None,
        fold: str = "left",
    ) -> SparseVector:
        """Alias for the classical left-fold moment-to-cumulant relation."""
        return self.moment_cumulant(variables, order=order, kind="classical", fold=fold)

    def moments(
        self,
        variables: Sequence[SparseVector],
        order: Optional[int] = None,
        fold: str = "left",
    ) -> SparseVector:
        """Alias for the classical left-fold cumulant-to-moment relation."""
        return self.cumulant_moment(variables, order=order, kind="classical", fold=fold)

    def logarithmic_generating_series(self, x: SparseVector, max_degree: Optional[int] = None) -> List[SparseVector]:
        """Compute the logarithm of the umbra-evaluated exponential, split by rank."""
        self.algebra_a._check_boundary()
        self.algebra_b._check_boundary()

        limit = min(self.algebra_a.max_degree, self.algebra_b.max_degree)
        if max_degree is not None:
            limit = min(limit, max_degree)

        mapped = self.umbra(self.algebra_a.exponential(x))
        logged = self.algebra_b.logarithm(mapped)

        result = [self.algebra_b.unit(0) for _ in range(limit + 1)]
        for key, coeff in logged._data.items():
            rank = self._rank_of(key)
            if rank is None:
                continue
            if 0 <= rank <= limit:
                result[rank][key] += coeff
        return result
