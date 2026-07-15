from __future__ import annotations

from fractions import Fraction
from typing import Any, Optional

from .core import SparseVector, coerce_basis_element


class Tensor(SparseVector):
    """A sparse tensor, viewed as a sparse vector over tuple-valued basis elements."""

    def __init__(self, data: Optional[SparseVector] = None):
        super().__init__()
        if data is None:
            return
        if isinstance(data, SparseVector):
            self._data.update({(key,): Fraction(value) for key, value in data._data.items() if Fraction(value) != 0})
            return
        raise TypeError("Tensor expects a SparseVector or None")

    @classmethod
    def unit(cls) -> "Tensor":
        result = cls()
        result[()] = 1
        return result

    def _coerce_key(self, key: Any) -> Any:
        if isinstance(key, tuple):
            return tuple(coerce_basis_element(item) for item in key)
        if isinstance(key, list):
            return tuple(coerce_basis_element(item) for item in key)
        return super()._coerce_key(key)

    def copy(self) -> "Tensor":
        result = Tensor()
        result._data.update({key: Fraction(value) for key, value in self._data.items() if Fraction(value) != 0})
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        normalized = self._coerce_key(key)
        super().__setitem__(normalized, value)

    def as_sparse_vector(self) -> SparseVector:
        """Return the underlying sparse vector representation."""
        return SparseVector({key: value for key, value in self._data.items()})

    def rank(self) -> int:
        if not self._data:
            return 0
        return max(len(key) for key in self._data)

    def min_rank(self) -> int:
        if not self._data:
            return 0
        return min(len(key) for key in self._data)

    def is_pure_of_rank(self, rank: int) -> bool:
        if not self._data:
            return rank == 0
        return all(len(key) == rank for key in self._data)

    def __str__(self) -> str:
        if not self._data:
            return "0"
        parts = []
        for key, coeff in sorted(self._data.items(), key=lambda item: str(item[0])):
            suffix = "⊗".join(str(part) for part in key)
            if coeff == 1:
                parts.append(suffix)
            elif coeff == -1:
                parts.append(f"-{suffix}")
            else:
                parts.append(f"{coeff}·{suffix}")
        return " + ".join(parts)

    def cat(self, other: Any) -> "Tensor":
        """Return the tensorial product of this tensor with another tensor-like object."""
        if isinstance(other, Tensor):
            result = Tensor()
            for left_key, left_coeff in self._data.items():
                for right_key, right_coeff in other._data.items():
                    result[left_key + right_key] += left_coeff * right_coeff
            return result
        if isinstance(other, SparseVector):
            return self.cat(tensor_from_element(other))
        raise TypeError("cat expects a Tensor or SparseVector")


def tensor(left: Optional[Any] = None, right: Optional[Any] = None) -> Tensor:
    """Create a tensor from one or two arguments.

    The API is intentionally conservative:
    - tensor(None, None) -> zero tensor
    - tensor(u, None) or tensor(None, u) -> rank-one tensor from a SparseVector
    - tensor(u, v) for two SparseVector objects -> tensor product
    - tensor(t, u) or tensor(u, t) for one Tensor and one SparseVector -> concatenation
    """
    if left is None and right is None:
        return Tensor()
    if left is None:
        return tensor_from_element(right)
    if right is None:
        return tensor_from_element(left)
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return left.cat(right)
    if isinstance(left, Tensor) and isinstance(right, SparseVector):
        return left.cat(tensor_from_element(right))
    if isinstance(left, SparseVector) and isinstance(right, Tensor):
        return tensor_from_element(left).cat(right)
    if isinstance(left, SparseVector) and isinstance(right, SparseVector):
        return tensor_from_element(left).cat(tensor_from_element(right))
    raise TypeError("tensor expects None, a Tensor, or a SparseVector")


def tensor_from_element(value: Any) -> Tensor:
    if value is None:
        return Tensor()
    if isinstance(value, Tensor):
        return value.copy()
    if isinstance(value, SparseVector):
        result = Tensor()
        for key, coeff in value._data.items():
            result[(key,)] = coeff
        return result
    raise TypeError("tensor_from_element expects None, a Tensor, or a SparseVector")
