"""Small shared helpers.

`expit` is written exactly as in the registered code (``1 / (1 + exp(-z))``), not as ``scipy.special.expit``.
The two differ in the last bits for some inputs, and the package must reproduce the registered numbers exactly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class ConvergenceWarning(UserWarning):
    """An iterative fit stopped at its iteration cap before meeting its convergence criterion."""


def expit(z):
    """Logistic function, ``1 / (1 + exp(-z))``, elementwise."""
    return 1 / (1 + np.exp(-z))


def warn_not_converged(what: str, maxit: int) -> None:
    # A constant message, so Python's default filter shows it once per call site rather than once per fit.
    warnings.warn(
        f"{what} reached its iteration cap (maxit={maxit}) before converging; "
        "inspect the result's `converged` attribute and consider raising `maxit`.",
        ConvergenceWarning,
        stacklevel=3,
    )


def as_float_array(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains missing or non-finite values")
    return a


def as_binary(x, name: str) -> np.ndarray:
    a = as_float_array(x, name)
    if not np.all((a == 0) | (a == 1)):
        raise ValueError(f"{name} must be coded 0/1")
    return a


def check_table(df: pd.DataFrame, outcome, exposure, covariates, outcome_complete=True):
    """Validate a gene table before any fit. Raises ValueError naming the offending column.

    ``exposure`` is one column name or a list of them; each must be complete, coded 0/1 (bool is accepted) and
    contain both exposed and unexposed genes. The outcome must be complete and coded 0/1; with
    ``outcome_complete=False`` (single exposure only) only the unexposed genes' outcomes are checked, because
    calibration and negative controls never read the exposed genes' outcomes. Covariates must be finite.
    """
    exposures = [exposure] if isinstance(exposure, str) else list(exposure)
    if not outcome_complete and len(exposures) != 1:
        raise ValueError("outcome_complete=False needs a single exposure")
    for c in [outcome, *exposures, *covariates]:
        if c not in df:
            raise ValueError(f"column {c!r} not in the table")
    for x in exposures:
        e = pd.to_numeric(df[x], errors="coerce").values.astype(float)
        if np.isnan(e).any() or not np.all((e == 0) | (e == 1)):
            raise ValueError(f"exposure {x!r} must be complete and coded 0/1")
        if e.sum() == 0 or e.sum() == len(e):
            raise ValueError(f"exposure {x!r} has no exposed or no unexposed genes")
    y = pd.to_numeric(df[outcome], errors="coerce").values.astype(float)
    if not outcome_complete:
        y = y[pd.to_numeric(df[exposures[0]], errors="coerce").values.astype(float) == 0]
    if np.isnan(y).any() or not np.all((y == 0) | (y == 1)):
        raise ValueError(f"outcome {outcome!r} must be coded 0/1"
                         + ("" if outcome_complete else " for the unexposed genes") + " with no missing values")
    for c in covariates:
        v = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        if not np.all(np.isfinite(v)):
            raise ValueError(f"covariate {c!r} has missing or non-finite values")
