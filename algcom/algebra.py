# algebra.py
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Sequence

from .core import SparseVector


def _unwrap(key):
    return key.value if hasattr(key, "value") else key


def lift_product(rule: Callable[[object, object], object]) -> Callable[[SparseVector, SparseVector], SparseVector]:
    """
    Turn a combinatorial product rule(a, b) -> SparseVector (or plain
    iterable of items, coefficient 1 each) on two BASIS elements into a
    bilinear map SparseVector x SparseVector -> SparseVector.
    """
    def lifted(left: SparseVector, right: SparseVector) -> SparseVector:
        result = SparseVector()
        for lk, lc in left._data.items():
            for rk, rc in right._data.items():
                outcome = rule(_unwrap(lk), _unwrap(rk))
                if isinstance(outcome, SparseVector):
                    for item, coeff in outcome._data.items():
                        result[item] += lc * rc * coeff
                else:
                    for item in outcome:
                        result[item] += lc * rc
        return type(left)(result)
    return lifted

def lift_unit(one: SparseVector) -> Callable[[Fraction],SparseVector]:
    '''
    Given an element '1' in the vector space A, creates the unit 
    function u:Q -> A given by u(r) = 1r
    '''
    if one is None : return None
    def lifted(scalar: Fraction) -> SparseVector : 
        return type(one)(one * scalar)
    return lifted

class Algebra:
    """Holds a (already-lifted) bilinear product and its unit."""

    def __init__(self, product: Callable[[SparseVector, SparseVector], SparseVector],
                 unit: Callable[[Fraction],SparseVector] | None = None):
        self.product = product
        self.unit = unit 

    @classmethod
    def from_rule(cls, rule, one: SparseVector | None = None) -> "Algebra":
        return cls(lift_product(rule), unit=lift_unit(one)) 

    def m(self, left: SparseVector, right: SparseVector) -> SparseVector:
        return self.product(left, right)

    def fold_left(self, elements: Sequence[SparseVector]) -> SparseVector:
        if not elements:
            return self.unit(1)
        result = elements[0]
        for e in elements[1:]:
            result = self.m(result, e)
        return result

    def fold_right(self, elements: Sequence[SparseVector]) -> SparseVector:
        if not elements:
            return self.unit(1)
        result = elements[-1]
        for e in reversed(elements[:-1]):
            result = self.m(e, result)
        return result

    def commutator(self, left: SparseVector, right: SparseVector) -> SparseVector:
        return self.m(left, right) - self.m(right, left)

    def power(self, x: SparseVector, n: int) -> SparseVector:
        if n == 0:
            return self.unit(1)
        result = x.copy()
        for _ in range(n - 1):
            result = self.m(result, x)
        return result

    def exponential(self, x: SparseVector, order: int = 5) -> SparseVector:
        """exp(x) = sum_{n=0}^{order} x^n / n! """
        result = self.unit(1)
        term = self.unit(1)
        fact = 1
        for n in range(1, order + 1):
            term = self.m(term, x)
            fact *= n
            result += term * Fraction(1, fact)
        return result

    def logarithm(self, x: SparseVector, order: int = 5) -> SparseVector:
        """log(x) = sum_{n=1}^{order} (-1)^{n+1} (x-1)^n / n"""
        result = self.unit(0)
        term = self.unit(1)
        x_ = x - self.unit(1)
        for n in range(1, order + 1):
            term = self.m(term, x_ )
            sign = 1 if n % 2 == 1 else -1
            result += term * Fraction(sign, n)
        return result

    def geometric(self, x: SparseVector, order: int = 5) -> SparseVector:
        """(1-x)^-1 truncated: sum_{n=0}^{order} x^n """
        result = self.unit(1)
        term = self.unit(1)
        for n in range(1, order + 1):
            term = self.m(term, x)
            result += term
        return result

    def rota_baxter(self, 
            R : Callable[[SparseVector],SparseVector],
            a : SparseVector,
            b : SparseVector) -> Fraction :
        '''
        Compute the parameter p in the expression
        R(a)R(b) - R(aR(b) + R(a)b) =  p ab
        If p does not exist, return None.
        '''
        Ra, Rb = R(a) , R(b)
        a_Rb = self.m(a,Rb)
        Ra_b = self.m(Ra,b)
        Left = self.m(Ra,Rb) - R( a_Rb + Ra_b)
        Right = R( self.m(a,b) )
        return Left.projection(Right) if Left.iscolinear(Right) else None
    
    def rota_leibnitz(self,
            D : Callable[[SparseVector],SparseVector],
            x : SparseVector,
            y : SparseVector) -> Fraction :
        '''
        Compute the parameter p in the expression
        D(xy) - D(x)y - xD(y) = p D(x)D(y)
        If p does not exist, return None.
        '''
        Dx , Dy = D(x) , D(y)
        Dx_y = self.m(Dx,y)
        x_Dy = self.m(x,Dy)
        Left = D( self.m(x,y) ) - Dx_y - x_Dy 
        Right = self.m(Dx,Dy)
        return Left.projection(Right) if Left.iscolinear(Right) else None
    