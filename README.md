
> algcom is MIT-licensed and free to use. If it's useful in your research, please cite it.

This package uses polymorphism and a lightweight algebraic abstraction to represent sparse linear combinations, tensors, and algebraic operations.

The current implementation provides a coherent toolkit for:

- SparseVector for sparse linear combinations of basis elements.
- Algebra for defining products and using generic operations such as powers, geometric series, exponentials, and logarithms.
- Series objects for ordinary and exponential generating functions.
- Tensor and sparse-tensor structures for multilinear and tensor-based constructions.

All of these modules are functional and are covered by an accompanying regression suite.


## Installation

One could install the package with pip.
```bash
pip install git+https://github.com/diegocaudillo/algcom
```

Read [how to run the notebooks](Notebooks.md) instead for first time users. 

## Quick start

```python
from algcom import SparseVector
from algcom.series import OrdinarySeries, OrdinaryTerm

v = SparseVector({"x": 2, "y": -1})
print(v)

left = SparseVector(OrdinaryTerm(1))
right = SparseVector(OrdinaryTerm(2))
print(OrdinarySeries.multiply(left, right))
```

This example uses the core sparse-vector type and a built-in ordinary-series algebra instance that is already defined in the package.

## Test suite

A regression suite covering the core vector, algebra, series, and tensor modules is available under the tests directory.

Run it locally with:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The suite exercises the public behaviors used by the package examples and the current implementation semantics, and it is intended to help keep the package stable as new algebraic constructions are added.

## Notebooks

The repository includes notebooks that are meant to be read as guided examples.

This is the current beginner's guide.

2. [notebooks/example_sq_matrices.ipynb](notebooks/example_sq_matrices.ipynb)
   - A worked example for building a new algebra.
   - Uses NumPy and 2x2 matrices as the guiding example.
   - Shows how to define a product, a unit, and use the generic algebra operations.
   - Includes the computation of exponential and other series.
   - Demonstrates duality in both finitely generated and full-dual settings.

## How to Cite

If you use `algcom` in academic work, please cite the software.

BibTeX:

```bibtex
@software{caudillo_algcom,
  author  = {Caudillo, Diego},
  title   = {algcom: A Python package for sparse linear combinations,
             tensors, and combinatorial algebra},
  year    = {2026},
  url     = {https://github.com/diegocaudillo/algcom},
  note    = {Version 0.x, accessed YYYY-MM-DD}
}
```

Plain text:

> Caudillo, D. (2026). *algcom: A Python package for sparse linear
> combinations, tensors, and combinatorial algebra* [Computer software].
> https://github.com/diegocaudillo/algcom


