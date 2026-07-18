# Running the notebooks locally

These instructions set up a clean environment to run the `algcom` example
notebooks on your own machine.

## 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Activate it:

- macOS / Linux:
```bash
  source venv/bin/activate
```
- Windows:
```bash
  venv\Scripts\activate
```

Your shell prompt should now show `(venv)` at the start of the line.

## 2. Install the required packages

```bash
pip install jupyter numpy
pip install git+https://github.com/diegocaudillo/algcom
```

## 3. Verify the install

```bash
python -c "from algcom import SparseVector; print(SparseVector())"
```

This should print: 
> 0

## 4. Download a notebook

`pip install` only installs the package, not the example notebooks. Fetch a
notebook directly from GitHub using its **raw** URL:

```bash
curl https://raw.githubusercontent.com/diegocaudillo/algcom/main/notebooks/example_sq_matrices.ipynb -o example_sq_matrices.ipynb
```

(Swap in the filename of any other notebook from the
[`notebooks/`](https://github.com/diegocaudillo/algcom/tree/main/notebooks)
folder the same way.)

Alternatively, clone the whole repository to get all notebooks at once:

```bash
git clone https://github.com/diegocaudillo/algcom.git
cd algcom
```

## 5. Launch Jupyter

From the same directory as the downloaded notebook (with `venv` still
active):

```bash
jupyter notebook
```

This opens a browser tab with a file listing. Click the `.ipynb` file to
open it.

## Troubleshooting: wrong kernel

If a notebook cell fails with `ModuleNotFoundError: No module named 'algcom'`
even though step 3 worked, Jupyter may be using a different Python kernel
than your `venv`. Check inside a notebook cell:

```python
import sys
print(sys.executable)
```

The printed path should point inside your `venv` folder. If it doesn't,
register the venv as its own kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name=algcom-venv --display-name "Python (algcom)"
```

Then, in Jupyter, use the **Kernel → Change Kernel** menu to select
"Python (algcom)".
