from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Sequence

from .core import SparseVector, BasisElement, coerce_basis_element

class TensorWord(BasisElement):
    '''An element of the basis of the tensor space.'''
    def __init__(self, value : Sequence) :
        if not isinstance(value,Sequence) :
            return None
        self.value = ( coerce_basis_element(v) for v in value )
        self._rank = len(self.value) 
    def __str__(self):
        return '⊗'.join(self.value)
