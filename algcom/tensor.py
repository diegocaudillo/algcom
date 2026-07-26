from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Union, Optional, Sequence

from .core import SparseVector, BasisElement, coerce_basis_element

class TensorWord(BasisElement):
    '''An element of the basis of the tensor space.'''

    @classmethod
    def _flattening(cls, items) :
        '''
        Assures the associativity of the tensor product by 
        flattening nested tensors. 
        It only works at one level of depth.
        '''
        ret = list()
        for e in items : 
            if isinstance(e,cls) :
                ret.extend(e.value)
            else :
                ret.append( coerce_basis_element(e) )
        return tuple(ret) 
            

    def __init__(self, value : Sequence) :
        if not isinstance(value,Sequence) : return 
        self.value = TensorWord._flattening(value)
        self.rank = len(self.value)
    
    def isfunction(self) -> bool: 
        return all(isinstance(f,Callable) for f in self.value)

    def pre_compose(self, word_of_functions : "TensorWord" ) -> "TensorWord":
        '''Evaluate a tensor of functions in this term entry-wise.'''
        if word_of_functions.rank != self.rank or not word_of_functions.isfunction():
            return None
        return TensorWord( tuple( 
            word_of_functions[i]( self.value[i] ) for i in range(self.rank)
            ) )
    
    def evaluate_at(self, point : "TensorWord") -> "TensorWord":
        if not self.isfunction() or self.rank != point.rank :
            return None
        return TensorWord( tuple(
            self.value[i]( point[i] ) for i in range(self.rank) 
        )) 

    @classmethod
    def power(cls, item , n : int) :  
        return cls(tuple( item for _ in range(n) ))

    def __str__(self):
        return '⊗'.join(str(v) for v in self.value)

class SparseTensor(SparseVector) : 
    '''
    Designed to carry on with the multi-linear structure of 
    the Tensor Gebra in combinatorial algebra.

    Each instance is a linear combination of TensorWord which the are
    homogenous items of the linear basis.
    '''
    def __init__(self, data : Optional[Union[
                    "SparseTensor", "SparseVector", list,
                    TensorWord, tuple , Any
                ]]= None):
        if data is None:
            super().__init__({})
            return
        if isinstance(data,SparseTensor) : 
            super().__init__(data)
        elif isinstance(data,SparseVector):
            super().__init__({ TensorWord( (k,) ) : v for k,v in data._data.items() })
        elif isinstance(data,list): 
            super().__init__({ k : 1 for k in data})
        elif isinstance(data,(TensorWord,tuple)) : 
            super().__init__({ TensorWord(data) : 1 })
        else :
            super().__init__({ TensorWord((data,)) : 1 })
    
    def min_rank(self) -> int:
        return min( w.rank for w in self._data.keys() )
    
    def max_rank(self) -> int:
        return max( w.rank for w in self._data.keys() )

    @classmethod
    def unit(cls,scalar: Fraction = 1) :
        return cls( SparseVector({ TensorWord() : scalar }) )

    @classmethod
    def bilinear(cls, left : SparseVector , righ : SparseVector) -> SparseTensor :
        result = SparseTensor()
        for lk, lc in SparseTensor(left)._data.items() : 
            for rk , rc in SparseTensor(righ)._data.items() : 
                result[ TensorWord( lk.value + rk.value ) ] += lc * rc
        return result
    
    @classmethod
    def multilinear(cls, items : Sequence[SparseVector]) : 
        if len(items) == 0 : return cls.unit()
        if len(items) == 1 : return cls( items[0] )
        result = cls.bilinear( items[0] , items[1] )
        for i in range(2,len(items)) : 
            result = cls.bilinear( result , items[i] )
        return result
