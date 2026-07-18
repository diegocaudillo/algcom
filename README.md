
> algcom is MIT-licensed and free to use. If it's useful in your research, please cite it.

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

The repository includes notebooks that are meant to be read as guided examples.

For the time being, this is the beginner's guide.

2. [notebooks/example_sq_matrices.ipynb](notebooks/example_sq_matrices.ipynb)
   - A more advanced walkthrough for building a new algebra.
   - Uses NumPy and 2x2 matrices as the guiding example.
   - Shows how to define a product, a unit, and use the generic algebra operations.
   - Includes the computation of exponential and other series
   - Examples of duality both finitely-generated and full-dual.

## How to Cite

If you use `algcom` in academic work, please cite both the software.

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


