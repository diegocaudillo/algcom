#polynomial.py

from .core import SparseVector
from .algebra import Algebra

from fractions import Fraction
from typing import Callable

class Polynomial(SparseVector) : 
    '''
    Class for one-parametric sparse polynomials
    '''
    @classmethod
    def one(cls):
        return cls({0:1})
    
    _Algebra = None
    @classmethod
    def algebra(cls) : 
        if cls._Algebra == None :
            pol_prod = lambda n,m : [n+m]
            cls._Algebra = Algebra.from_rule( pol_prod , one=cls.one() )
        return cls._Algebra

    @classmethod
    def e_sub_a_functional(cls,a : Fraction) -> Callable[["Polynomial"],Fraction]: 
        def fun(pol : "Polynomial") -> Fraction : 
            result = Fraction(0)
            for n,c in pol._data.items() : 
                result += c * Fraction(a**int(n.value))
            return result
        return fun


    def multiply(self, other : "Polynomial") -> "Polynomial" : 
        return Polynomial.algebra().m(self,other)

    def evaluate_at(self, a: Fraction) -> Fraction :
        return Polynomial.e_sub_a_functional(a)(self)

    def __str__(self):
        tmp = SparseVector({f"x^{n}": c for n,c in self._data.items()})
        tmp[1]=tmp["x^0"]
        tmp["x"] = tmp["x^1"]
        tmp["x^1"] = tmp["x^0"]=0
        return tmp.__str__() 
    
def _poly_little_test():
    px = Polynomial({0:-1, 1:0, 2:1})
    q1 = Polynomial({0:1, 1:1})
    q2 = Polynomial({0:-1, 1:1})

    q = q1.multiply(q2)
    test1 = "OK" if q == px else "FAIL"
    print(f"Factorisation: {q1} * {q2} = {px} [{test1}]") 
    print("")
    print(f"Evaluation p(-1)= {px.evaluate_at(-1)}")
    print(f"Evaluation p( 0)={px.evaluate_at(0)}")
    print(f"Evaluation p( 1)= {px.evaluate_at(1)}")
 
    npx = Polynomial({0:1,1:1}) # 1 + x
    print(f"log{npx} = {Polynomial.algebra().logarithm(npx)}") 
    
