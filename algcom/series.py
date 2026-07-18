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


_degree = lambda n : n 


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
    
    def __str__(self) -> str:
        return _series_str(self, factorial_weight=True)


def _series_str(series: SparseVector, factorial_weight: bool) -> str:
    terms = []
    for key, coeff in sorted(series._data.items(), key=lambda item: _degree(item[0])):
        n = _degree(key)
        c = round(float(coeff),2) if coeff.denominator > 1000 else coeff
        label = "1" if n == 0 else (f"t" if n == 1 else f"t^{n}")
        if factorial_weight and n > 1:
            label += f"/{n}!"
        terms.append(f"({c}){'' if label == '1' else '*' + label}" if label != "1" else f"({coeff})")
    return " + ".join(terms) if terms else "0"


def _little_test() : 
    import numpy as np 
    A = ExponentialSeries.algebra()

    X = np.random.normal(size=1000000) 
    M = ExponentialSeries( { n : np.mean(X**n) for n in range(7)  } )

    print(f"MGF_X(t) = ", M )
    print(f"K_X(t) = ",A.logarithm(M,deg=_degree) )

    X =  X + np.random.normal(size=1000000) 
    M = ExponentialSeries( { n : np.mean(X**n) for n in range(7)  } )
    
    print(f"MGF_(X+Y)(t) = ", M )
    print(f"K_(X+Y)(t) = ",A.logarithm(M,deg=_degree) )
    