from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Any, Dict, Iterable, Iterator, MutableMapping, Optional, Union


class ZeroType:
    def __repr__(self) -> str:
        return "Zero"

    def __str__(self) -> str:
        return "0"


Zero = ZeroType()


class BasisElement:
    """A lightweight wrapper around a user basis element."""

    def __init__(self, value: Any):
        self.value = value

    def copy(self) -> "BasisElement":
        return BasisElement(self.value)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BasisElement):
            return self.value == other.value
        return self.value == other

    def __hash__(self) -> int:
        return hash((self.__class__, self.value))

    def __repr__(self) -> str:
        return f"BasisElement({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, BasisElement):
            return str(self) < str(other)
        return str(self) < str(other)

    def rank(self) -> Optional[int]:
        method = getattr(self.value, "rank", None)
        if callable(method):
            return method()
        return None

    def degree(self) -> Optional[int]:
        method = getattr(self.value, "degree", None)
        if callable(method):
            return method()
        return None

    def weight(self) -> Optional[int]:
        method = getattr(self.value, "weight", None)
        if callable(method):
            return method()
        return None

    def __len__(self) -> int:
        try:
            return len(self.value)
        except TypeError:
            return 0


def coerce_basis_element(value: Any) -> Any:
    if isinstance(value, BasisElement):
        return value
    return BasisElement(value)


class SparseVector(MutableMapping[Any, Fraction]):
    """Dictionary-backed sparse vector over a set of basis elements."""

    def __init__(self, data: Optional[Union["SparseVector", Dict[Any, Any], Iterable[Any], Any]] = None):
        self._data: Dict[Any, Fraction] = {}
        if data is None:
            return
        if isinstance(data, SparseVector):
            self._data.update({k: Fraction(v) for k, v in data._data.items() if v != 0})
            return
        if isinstance(data, dict):
            for key, value in data.items():
                self[key] = value
            return
        if isinstance(data, (tuple, list)):
            if len(data) == 0:
                return
            if len(data) == 1:
                self[data[0]] = 1
                return
            raise TypeError("Expected a single element or a mapping for initialization")
        self[data] = 1

    def _coerce_key(self, key: Any) -> Any:
        return coerce_basis_element(key)

    def __getitem__(self, key: Any) -> Fraction:
        return self._data.get(self._coerce_key(key), Fraction(0))

    def __setitem__(self, key: Any, value: Any) -> None:
        coeff = Fraction(value)
        normalized = self._coerce_key(key)
        if coeff == 0:
            self._data.pop(normalized, None)
            return
        self._data[normalized] = coeff

    def __delitem__(self, key: Any) -> None:
        del self._data[self._coerce_key(key)]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def copy(self) -> "SparseVector":
        return SparseVector(self)

    def __add__(self, other: Any) -> "SparseVector":
        if other is Zero:
            return self.copy()
        if isinstance(other, SparseVector):
            result = self.copy()
            for key, coeff in other._data.items():
                result[key] += coeff
            return result
        return NotImplemented

    def __sub__(self, other: Any) -> "SparseVector":
        if other is Zero:
            return self.copy()
        if isinstance(other, SparseVector):
            result = self.copy()
            for key, coeff in other._data.items():
                result[key] -= coeff
            return result
        return NotImplemented

    def __iadd__(self, other: Any) -> "SparseVector":
        if other is Zero:
            return self
        if isinstance(other, SparseVector):
            for key, coeff in other._data.items():
                self[key] += coeff
            return self
        return NotImplemented

    def __isub__(self, other: Any) -> "SparseVector":
        if other is Zero:
            return self
        if isinstance(other, SparseVector):
            for key, coeff in other._data.items():
                self[key] -= coeff
            return self
        return NotImplemented

    def __mul__(self, other: Any) -> "SparseVector":
        if isinstance(other, (int, Fraction)):
            result = SparseVector()
            for key, coeff in self._data.items():
                result[key] = coeff * other
            return result
        return NotImplemented

    def __rmul__(self, other: Any) -> "SparseVector":
        return self.__mul__(other)

    def __imul__(self, other: Any) -> "SparseVector":
        if isinstance(other, (int, Fraction)):
            for key in list(self._data):
                self[key] *= other
            return self
        return NotImplemented

    def __neg__(self) -> "SparseVector":
        return -1 * self

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SparseVector):
            return self._data == other._data
        return False

    def __hash__(self) -> int:
        return hash(tuple(sorted((key, coeff) for key, coeff in self._data.items())))

    def bracket(self, other: "SparseVector") -> Fraction:
        """Return the Kronecker pairing between this vector and another."""
        result = Fraction(0)
        for key, coeff in self._data.items():
            result += coeff * other._data.get(key, Fraction(0))
        return result

    def dual(self) -> Any:
        """Return a callable dual functional for this sparse vector."""
        def functional(other: "SparseVector") -> Fraction:
            return self.bracket(other)

        return functional

    def __str__(self) -> str:
        if not self._data:
            return "0"

        terms = []
        for key, coeff in sorted(self._data.items(), key=lambda item: str(item[0])):
            denominator = coeff.denominator
            numerator = coeff.numerator
            if numerator == 0:
                continue
            if denominator != 1:
                g = gcd(abs(numerator), denominator)
                numerator //= g
                denominator //= g
            if numerator == 1 and denominator == 1:
                term = str(key)
            elif numerator == -1 and denominator == 1:
                term = f"-{str(key)}"
            elif denominator == 1:
                term = f"{numerator} {str(key)}"
            else:
                term = f"{numerator}/{denominator} {str(key)}"
            terms.append(term)

        if not terms:
            return "0"

        parts = []
        for idx, term in enumerate(terms):
            if idx == 0:
                if term.startswith("-"):
                    parts.append(term)
                else:
                    parts.append(term)
            else:
                if term.startswith("-"):
                    parts.append("- " + term[1:])
                else:
                    parts.append("+ " + term)
        return " ".join(parts)
