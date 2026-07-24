from __future__ import annotations

from fractions import Fraction
from math import lcm, sqrt 
from typing import Any, Dict, Iterable, Iterator, MutableMapping, Optional, Union

from functools import reduce

Id = lambda x : x

class ZeroType:
    def __repr__(self) -> str:
        return "Zero"

    def __str__(self) -> str:
        return "0"


Zero = ZeroType()

def _has_copy(object) -> bool:
    return hasattr(object, 'copy') and callable(getattr(object, 'copy'))


class BasisElement:
    """A lightweight wrapper around a user basis element."""

    def __init__(self, value: Any):
        self.value = value.copy() if _has_copy(value) else value

    def copy(self) -> "BasisElement":
        return BasisElement(self.value)
    
    def _order_key(self) : 
        if hasattr(self.value,"__lt__") : 
            return self.value
        return self.__str__() 

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
        return str(self) < str(other)


def coerce_basis_element(value: Any) -> Any:
    if isinstance(value, BasisElement):
        return value
    return BasisElement(value)


class SparseVector(MutableMapping[Any, Fraction]):
    """Dictionary-backed sparse vector over a set of basis elements."""

    _tol = Fraction(1e-3).limit_denominator() 
    _lim = 1000

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
        if isinstance(data, list):
            self._data.update({k: Fraction(1) for k in data})
            return
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
            return type(self)(result)
        return NotImplemented

    def __sub__(self, other: Any) -> "SparseVector":
        if other is Zero:
            return self.copy()
        if isinstance(other, SparseVector):
            result = self.copy()
            for key, coeff in other._data.items():
                result[key] -= coeff
            return type(self)(result)
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
            return type(self)(result)
        return NotImplemented

    def __rmul__(self, other: Any) -> "SparseVector":
        return type(self)(self.__mul__(other))

    def __imul__(self, other: Any) -> "SparseVector":
        if isinstance(other, (int, Fraction)):
            for key in list(self._data):
                self[key] *= other
            return self
        return NotImplemented

    def __neg__(self) -> "SparseVector":
        return type(self)(-1 * self)

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
        return result.limit_denominator(self._lim)

    def dual(self) -> Any:
        """Return a callable dual functional for this sparse vector."""
        def functional(other: "SparseVector") -> Fraction:
            return self.bracket(other)

        return functional
    
    def cosine(self, other : "SparseVector") -> float : 
        if self is Zero or other is Zero :
            return 0
        dot = self.bracket(other)
        left = self.bracket(self)
        right = other.bracket(other) 
        return dot / ( sqrt( left * right) ) 
    
    def iscolinear(self, other: "SparseVector" ) -> bool :
        return abs( abs(self.cosine(other)) - 1.0 ) < type(self)._tol
    
    def projection(self, other: "SparseVector") -> Fraction : 
        norm_sq = self.bracket(self)
        if norm_sq < type(self)._tol : 
            return 0
        return self.bracket(other) / norm_sq 

    def integer_coefficients(self):
        """
        Returns (d, obj) where d is the LCM of all coefficient denominators
        (an int), and obj is a SparseVector with every coefficient equal to
        d times the original (hence integral, denominator == 1).
        """
        if not self._data:
            return 1, SparseVector()

        denominators = (c.denominator for c in self._data.values())
        d = reduce(lcm, denominators, 1)

        result = SparseVector()
        for key, coeff in self._data.items():
            result._data[key] = Fraction(coeff * d)

        return d, result 

    def _decimal(self) : 
        terms = []
        
        for key, coeff in self._ordered_items():
            c = round( float(coeff) , 2) 
            if abs(c) < max(0.01,self._tol):
                continue
            if abs(c-1.0) < max(0.01,self._tol):
                term = str(key)
            elif abs(c-1) < max(0.01,self._tol):
                term = f"-{str(key)}"
            else:
                term = f"{c} {str(key)}"
            terms.append(term)

        if not terms:
            return "0"

        parts = []
        for idx, term in enumerate(terms):
            if idx == 0:
                parts.append(term)
            else:
                if term.startswith("-"):
                    parts.append("- " + term[1:])
                else:
                    parts.append("+ " + term)

        return "⟅" + " ".join(parts) + "⟆"

    def _ordered_items(self) : 
        return sorted(self._data.items(), key=lambda item: item[0]._order_key())

    def __str__(self) -> str:
        if not self._data:
            return "0"

        terms = []

        d , integer_self = self.integer_coefficients()
        if d >= self._lim : return self._decimal () 

        for key, coeff in integer_self._ordered_items():
            numerator = coeff.numerator  # denominator is guaranteed to be 1 here
            if numerator == 0:
                continue
            if numerator == 1:
                term = str(key)
            elif numerator == -1:
                term = f"-{str(key)}"
            else:
                term = f"{numerator} {str(key)}"
            terms.append(term)

        if not terms:
            return "0"

        parts = []
        for idx, term in enumerate(terms):
            if idx == 0:
                parts.append(term)
            else:
                if term.startswith("-"):
                    parts.append("- " + term[1:])
                else:
                    parts.append("+ " + term)

        body = "⟅" + " ".join(parts) + "⟆"

        if d == 1:
            return f"{body}"
        return f"{body}/{d}"
