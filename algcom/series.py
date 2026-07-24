# series.py
"""
Formal power series in one variable, in the two flavours used throughout
algebraic combinatorics:

  - OrdinarySeries:     A(t) = sum a_n t^n,        product  t^n . t^m = t^{n+m}.
  - ExponentialSeries:  A(t) = sum a_n t^n/n!,      product  t^n . t^m = C(n+m,n) t^{n+m}.
"""

from __future__ import annotations
from math import comb

from .core import BasisElement, SparseVector
from .algebra import Algebra

class OrdinaryTerm(BasisElement) : 
    """Ordinary generating functions, basis t^n, plain convolution."""
    def __str__(self):
        if self.value == 0 : return "1_Q"
        if self.value == 1 : return "t" 
        return f"t^{self.value}"
    
OrdinarySeries = Algebra.from_rule(
    rule = lambda n,m : [OrdinaryTerm(n+m)], 
    one = SparseVector(OrdinaryTerm(0)),
    degree_method = lambda n : n  
)

class ExponentialTerm(BasisElement) : 
    """Exponential generating functions, A(t) = sum a_n t^n/n!."""
    def __str__(self):
        if self.value == 0 : return "1_Q"
        if self.value == 1 : return "t" 
        return f"t^{self.value}/{self.value}!"

ExponentialSeries = Algebra.from_rule(
    rule = lambda n, m: SparseVector({ExponentialTerm(n + m): comb(n + m, n)}) , 
    one = SparseVector(ExponentialTerm(0)),
    degree_method = lambda n : n 
)


def _exp_series_test(order = 7) : 
    import numpy as np 
    A = ExponentialSeries.copy() # Not a good idea to change the algebra globaly
    A.max_degree = order

    X = np.random.normal(size=1000000) 
    M = SparseVector( { ExponentialTerm(n) : np.mean(X**n) for n in range(order+1)  } )

    print(f"MGF_X(t) = ", M )
    print(f"K_X(t) = ",A.logarithm(M) )
    print("\n")

    X =  X + np.random.normal(size=1000000) 
    M = SparseVector( { ExponentialTerm(n) : np.mean(X**n) for n in range(order+1)  } )
    
    print(f"MGF_(X+Y)(t) = ", M )
    print(f"K_(X+Y)(t) = ",A.logarithm(M) )

def _poly_test() : 
    t0 = SparseVector(OrdinaryTerm(0))
    t1 = SparseVector(OrdinaryTerm(1))
    t2 = SparseVector(OrdinaryTerm(2))

    q = t2 - t0 
    p1 = t1 - t0 
    p2 = t1 + t0 
    p = OrdinarySeries.multiply(p1,p2)

    test1 = "OK" if q == p else "FAIL"

    print(f"Factorisation: {p1} * {p2} = {p} [{test1}]") 
    print("")
