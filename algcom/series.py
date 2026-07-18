# series.py
"""
Formal power series in one variable, in the two flavours used throughout
algebraic combinatorics:

  - OrdinarySeries:     A(t) = sum a_n t^n,        product  t^n . t^m = t^{n+m}.
  - ExponentialSeries:  A(t) = sum a_n t^n/n!,      product  t^n . t^m = C(n+m,n) t^{n+m}.

Both are built the same way `Polynomial` is: a basis rule on non-negative
integers, lifted to a bilinear product via `Algebra.from_rule`. Once the
product and the unit are given, `Algebra.exponential`/`Algebra.logarithm`
(already generic, see algebra.py) compute the *-exponential and *-logarithm
of a series with no further code -- these are exactly the classical
moment-cumulant / cumulant-moment formulas of 20xHopfAlgebras.tex
("Algebraic cumulants") and mathdef.tex/main.tex ("Algebraic Probability")
when applied to ExponentialSeries.

The rule for ExponentialSeries is the Cauchy convolution
    (a*b)_n = sum_k C(n,k) a_k b_{n-k},
i.e. `Delta_shuffle`'s dual product; see notebooks/series_duality.ipynb for
the derivation of this same rule directly from the coproduct, as a check.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Callable, Dict

from .core import SparseVector
from .algebra import Algebra


def _degree(key) -> int:
    return key.value if hasattr(key, "value") else key


class OrdinarySeries(SparseVector):
    """Ordinary generating functions, basis t^n, plain convolution."""

    @classmethod
    def one(cls) -> "OrdinarySeries":
        return cls({0: 1})

    _Algebra = None

    @classmethod
    def algebra(cls) -> Algebra:
        if cls._Algebra is None:
            rule = lambda n, m: [n + m]
            cls._Algebra = Algebra.from_rule(rule, one=cls.one())
        return cls._Algebra

    def multiply(self, other: "OrdinarySeries") -> "OrdinarySeries":
        return OrdinarySeries.algebra().m(self, other)

    def coefficient(self, n: int) -> Fraction:
        return self[n]

    def truncate(self, order: int) -> "OrdinarySeries":
        return type(self)({_degree(k): c for k, c in self._data.items() if _degree(k) <= order})

    def __str__(self) -> str:
        return _series_str(self, factorial_weight=False)


class ExponentialSeries(SparseVector):
    """
    Exponential generating functions, basis t^n, tracked by RAW
    coefficients a_n (i.e. A(t) = sum a_n t^n/n!). Product is Cauchy's
    convolution, weighted by binomial coefficients.
    """

    @classmethod
    def one(cls) -> "ExponentialSeries":
        return cls({0: 1})

    _Algebra = None

    @classmethod
    def algebra(cls) -> Algebra:
        if cls._Algebra is None:
            rule = lambda n, m: SparseVector({n + m: comb(n + m, n)})
            cls._Algebra = Algebra.from_rule(rule, one=cls.one())
        return cls._Algebra

    def multiply(self, other: "ExponentialSeries") -> "ExponentialSeries":
        return ExponentialSeries.algebra().m(self, other)

    @classmethod
    def from_moments(cls, moments: Dict[int, Fraction]) -> "ExponentialSeries":
        """`moments` are the raw moments m_n = E[X^n]; m_0 = 1 is assumed
        if not given explicitly."""
        data = dict(moments)
        data.setdefault(0, 1)
        return cls(data)

    def moment(self, n: int) -> Fraction:
        return self[n]

    def cumulants(self, degree: int = 5) -> "ExponentialSeries":
        """The *-logarithm of this series, i.e. the classical
        cumulant-moment formula (eq.class.CM in 20xHopfAlgebras.tex).

        `degree` is the highest power of t trusted in the result. Internally,
        Algebra.logarithm's convolution-power count is set to `degree` too
        (that many self-products of (M-1) are exactly what is needed to
        reach degree `degree`, no more, no less) and everything the
        convolution reaches *beyond* `degree` is discarded: those terms
        implicitly assume all moments past `degree` vanish, which is false,
        so they are not trustworthy cumulants and must not be kept.
        """
        K = ExponentialSeries.algebra().logarithm(self, order=degree)
        return K.truncate(degree)

    @classmethod
    def from_cumulants(cls, cumulants: Dict[int, Fraction], degree: int = 5) -> "ExponentialSeries":
        """The *-exponential of a cumulant series, i.e. the classical
        moment-cumulant formula (eq.class.MC in 20xHopfAlgebras.tex).
        `cumulants` should have no n=0 term (k_0 = 0 by convention).
        See `cumulants` above for why `degree` truncates both the
        convolution-power count and the returned series."""
        K = cls(dict(cumulants))
        if K[0] != 0:
            raise ValueError("cumulant series must have k_0 = 0")
        M = ExponentialSeries.algebra().exponential(K, order=degree)
        return M.truncate(degree)

    def truncate(self, order: int) -> "ExponentialSeries":
        return type(self)({_degree(k): c for k, c in self._data.items() if _degree(k) <= order})

    def __str__(self) -> str:
        return _series_str(self, factorial_weight=True)


def _series_str(series: SparseVector, factorial_weight: bool) -> str:
    terms = []
    for key, coeff in sorted(series._data.items(), key=lambda item: _degree(item[0])):
        n = _degree(key)
        label = "1" if n == 0 else (f"t" if n == 1 else f"t^{n}")
        if factorial_weight and n > 1:
            label += f"/{n}!"
        terms.append(f"({coeff}){'' if label == '1' else '*' + label}" if label != "1" else f"({coeff})")
    return " + ".join(terms) if terms else "0"
