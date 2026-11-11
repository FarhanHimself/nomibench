"""Propensity score for a nomination, given discoverability covariates.

The registered model (paper 1, spec v1 section C): logistic regression of the exposure on the covariates, each
entered as a natural cubic regression spline with 3 df (patsy ``cr(x, df=3, constraints='center')``), except
covariates listed as linear and covariates whose 1/3 and 2/3 quantiles coincide, which enter linearly.

Diagnostics: the c-statistic (Mann-Whitney) and standardised mean differences before and after weighting by
propensity-score quintile (ATT-style quintile weights).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import mannwhitneyu

__all__ = ["spline_formula", "PropensityScore", "fit_ps", "c_stat", "balance"]


def spline_formula(df: pd.DataFrame, cols, linear=()):
    """patsy formula with a centred 3-df cubic regression spline per covariate (linear where requested or where
    the 1/3 and 2/3 quantiles coincide)."""
    parts = []
    for c in cols:
        q33, q67 = np.quantile(df[c], [1 / 3, 2 / 3])
        parts.append(c if (c in linear or q33 == q67) else f"cr({c}, df=3, constraints='center')")
    return " + ".join(parts)


@dataclass
class PropensityScore:
    ps: np.ndarray            # fitted P(exposure | covariates)
    logit: np.ndarray         # log(ps / (1 - ps)), the G1 adjustment variable
    formula: str
    converged: bool
    c_statistic: float

    def __iter__(self):       # legacy unpacking: ps, lps, formula = fit_ps(...)
        yield self.ps
        yield self.logit
        yield self.formula


def fit_ps(df: pd.DataFrame, exposure: str, cols, linear=(), extra_terms=(), maxiter=300) -> PropensityScore:
    """Fit the registered propensity model ``exposure ~ s(cols) + extra_terms`` by maximum likelihood.

    Parameters
    ----------
    df : one row per gene.
    exposure : 0/1 column.
    cols : discoverability covariates.
    linear : covariates entered linearly (paper 1: the number of contributing eQTL datasets, ``x_nd``).
    extra_terms : further patsy terms entered as written (e.g. missingness indicators).
    """
    f = spline_formula(df, cols, linear)
    if extra_terms:
        f += " + " + " + ".join(extra_terms)
    Xps = dmatrix(f, df, return_type="dataframe")
    if np.linalg.matrix_rank(Xps.values) != Xps.shape[1]:
        raise ValueError("propensity design is rank-deficient: " + f)
    e = df[exposure].astype(float).values
    fit = sm.Logit(e, Xps).fit(disp=0, maxiter=maxiter)
    ps = np.asarray(fit.predict(Xps))
    conv = bool(fit.mle_retvals.get("converged", True)) if hasattr(fit, "mle_retvals") else True
    with np.errstate(divide="ignore"):
        lps = np.log(ps / (1 - ps))
    if not np.all(np.isfinite(lps)):
        raise ValueError("the propensity model separates the exposure (fitted probabilities of 0 or 1); "
                         "G1 cannot adjust for it: " + f)
    return PropensityScore(ps, lps, f, conv, c_stat(ps, e))


def c_stat(ps, e):
    """Area under the ROC curve of the propensity score for the exposure (NaN if either group is empty)."""
    ps = np.asarray(ps, float)
    e = np.asarray(e, bool)
    if e.all() or not e.any():
        return float("nan")
    return float(mannwhitneyu(ps[e], ps[~e]).statistic / (e.sum() * (~e).sum()))


def balance(df: pd.DataFrame, exposure: str, cols, ps):
    """Standardised mean differences (exposed minus other, pooled SD) before and after weighting by
    propensity-score quintile. Returns ``(table, quintile labels)``."""
    psq = pd.qcut(ps, 5, labels=False, duplicates="drop")
    e = df[exposure].values.astype(bool)
    s = pd.Series(psq)
    ns_ = s.value_counts()
    n1 = s[e].value_counts()
    n0 = s[~e].value_counts()
    w = np.where(e, s.map(ns_ / n1), s.map(ns_ / n0)).astype(float)
    out = []
    for c in cols:
        a, b = df.loc[e, c], df.loc[~e, c]
        sd = np.sqrt((a.var() + b.var()) / 2)
        raw = (a.mean() - b.mean()) / sd
        adj = (np.average(a, weights=w[e]) - np.average(b, weights=w[~e])) / sd
        out.append(dict(covariate=c, smd_raw=raw, smd_psq=adj))
    return pd.DataFrame(out), psq
