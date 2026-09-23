# bayesian_utilities
A few helper functions for simple Bayesian analysis in python

## Overview

`bayesian_utilities` is a lightweight toolkit of helper functions supporting simple Bayesian 
analysis workflows. It currently provides two main areas of functionality:

- **General Bayesian utilities** — covariance estimation, grid/mesh helpers for evaluating 
  functions and posteriors, inverse-transform sampling, and visualization via scatterplot/contour 
  matrices.
- **Space-filling designs** — design-of-experiment sampling for surrogate modeling and emulation, 
  including Latin-hypercube, Sobol, and greedy-maximin / support-points sub-sampling methods.

## Installation

Requires Python >= 3.11 (see `pyproject.toml`).

```bash
pip install -e .
```

To include development dependencies (for running the test suite):

```bash
pip install -e ".[dev]"
```

> Note: the build backend is `uv_build` (see `pyproject.toml`), so you may also
> install/manage the environment with [uv](https://github.com/astral-sh/uv).

### Dependencies

`bayesian_utilites` is build on the basic python numerical software stack: `numpy`, `scipy`, `matplotlib`, and `numba`.

- The `main` branch depends on a recent version of `scipy>=1.17.0`,
- A `version_friendly` branch is maintained for older dependence on `scipy>=1.12.0` and also strips any dependence on `numpy`.