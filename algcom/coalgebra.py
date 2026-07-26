# coalgebra.py

from fractions import Fraction
from typing import Callable, Sequence, Any

from .core import SparseVector, LinearLift, Zero
from tensor import SparseTensor, TensorWord


class CoAlgebra:

    def __init__(self, coproduct: Callable[[SparseVector], SparseTensor],
                 counit: Callable[[SparseVector],Fraction] | None = None,
                 one: SparseVector | None = None):
        self.coproduct = coproduct
        self.counit = counit 
        self.one = one

    def copy(self) : 
        return CoAlgebra(self.coproduct,self.counit,self.one)

    @classmethod
    def from_cuts(cls, 
                  cuts, 
                  one : Callable[[SparseVector],Fraction] | None = None) -> "CoAlgebra":
        '''
        Create a coalgebra from combinatorial cuts. 
        A cuts(x) maps x to a list [ (x_1^i,x_2^i) ]_i.

        Example, for ordinary series cuts(4) = [(4,0),(3,1),(2,2),(1,3),(0,4)].
        The counit (if extant) is the (Kronecker) dual of the "one".
        '''
        rule = lambda x : SparseTensor( TensorWord(w) for w in cuts(x) )
        counit = SparseVector(one).dual() if one is not None else None
        return cls(LinearLift(rule),counit,one) 

    @classmethod
    def from_coefficients(cls,
                    catalog : Sequence,
                    coefficients : Callable[[Any,Any,Any],Fraction],

                    one: SparseVector | None = None, 
                    degree_method : Callable[[Any],int] = None,
                    max_degree: int | None =None) -> "CoAlgebra":

        validate = (
            lambda x : degree_method(x) <= max_degree 
                if degree_method is not None and max_degree is not None
                else True
            )
        def rule(a,b):
            res = SparseVector()
            for c in catalog : 
                if not validate(c) : continue
                res[c] = coefficients(a,b,c) 
            return res
        counit = SparseVector(one).dual() if one is not None else None
        return cls.from_rule(rule,counit,degree_method,max_degree,one)

    def reduced_coproduct(self, vector) :
        if self.one is None : return NotImplemented
        res = ( 
            self.coproduct(vector) 
            - SparseTensor.bilinear(self.one,vector) 
            - SparseTensor.bilinear(vector,self.one)
        )

    def is_primitive(self,vector : SparseVector) : 
        return self.reduced_coproduct(vector) is Zero

