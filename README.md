# Alert: Technical Description

This code uses polymorphism and a lightweight algebraic abstraction to represent sparse linear combinations, tensors, and algebraic operations. 

The package is currently running two main ideas:

- SparseVector for sparse linear combinations of basis elements.
- Algebra for defining a product and using generic operations such as powers, geometric series, exponentials, and logarithms.

## Stability notice

Only the following parts are considered stable releases:

- SparseVector
- Polynomial Algebra

The tensor machinery and other experimental pieces should be treated as work in progress.

## Installation

One could install the package with pip.
```bash
pip install git+https://github.com/diegocaudillo/algcom
```

## Quick start

```python
from algcom import SparseVector
from algcom.polynomial import Polynomial

v = SparseVector({"x": 2, "y": -1})
print(v)

p = Polynomial({0: -1, 2: 1})
print(p.evaluate_at(1))
```

## Notebooks

The repository includes two notebooks that are meant to be read as guided examples:

1. [notebooks/sparse_vectors_and_polynomials.ipynb](notebooks/sparse_vectors_and_polynomials.ipynb)
   - A simple introduction to sparse vectors.
   - A small example of the algebra of complex numbers using the algebra machinery.
   - A short polynomial example.

2. [notebooks/building_a_new_combinatorial_algebra.ipynb](notebooks/building_a_new_combinatorial_algebra.ipynb)
   - A more advanced walkthrough for building a new algebra.
   - Uses NumPy and 2x2 matrices as the guiding example.
   - Shows how to define a product, a unit, and use the generic algebra operations.
