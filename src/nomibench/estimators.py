"""Estimators of enrichment of an outcome (e.g. SMR/HEIDI support) among nominated genes.

G1 (primary)  Firth logistic regression of Y on the exposure and a 3-df cubic spline of the logit propensity score;
              penalised LRT P, profile penalised-likelihood CIs, Wald SE (paper 1 spec v1/v2).
G1b           Mantel-Haenszel OR across propensity-score quintiles (RBG variance).
G1c           Observed / expected events among the exposed, with the expectation from a Firth model fitted on the
              unexposed genes only; exact Poisson-binomial P.
H             Exposure x in-locus interaction in the G1 model.
W1            Conditional Firth logistic regression within GWAS loci (strata = locus intervals).
W2            Mantel-Haenszel by locus interval.
naive         Fisher's exact test, conditional MLE odds ratio.

All functions take NumPy arrays aligned gene by gene; the propensity score comes from
:func:`nomibench.propensity.fit_ps`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import chi2, norm

from ._util import expit
from .conditional import cfl_fit, cfl_profile_ci, make_strata
from .firth import DEFAULT_MAXIT, firth_fit, firth_lrt, firth_profile_ci, wald_se
from .tables import fisher, mh_rbg, poibin_two_sided, strat_counts

__all__ = ["g1_design", "h_design", "G1Result", "g1", "g1b", "g1c", "h_interaction", "W1Result", "w1", "w2",
           "naive", "unadjusted_firth", "Tuple4"]


def g1_design(lps, e):
    """Design ``[1, E, s(logit PS)]``: intercept, exposure, centred 3-df cubic regression spline of the logit PS."""
    B = dmatrix("cr(v, df=3, constraints='center')", pd.DataFrame(dict(v=lps)), return_type="dataframe").values
    return np.c_[B[:, :1], np.asarray(e, float), B[:, 1:]]


def h_design(lps, e, L):
    """Design ``[1, E, L, E:L, s(logit PS)]`` for the exposure x in-locus interaction."""
    B = dmatrix("cr(v, df=3, constraints='center')", pd.DataFrame(dict(v=lps)), return_type="dataframe").values
    e = np.asarray(e, float)
    L = np.asarray(L, float)
    return np.c_[B[:, :1], e, L, e * L, B[:, 1:]]


@dataclass
class G1Result:
    """G1 estimate on the log-odds scale; ``odds_ratio`` and the ``*_or`` helpers exponentiate."""
    log_or: float
    se: float                       # Wald SE from the expected information at the Firth estimate
    pvalue: float                   # penalised likelihood-ratio test
    profile: dict = field(default_factory=dict)     # level -> (lo, hi), log scale
    wald: dict = field(default_factory=dict)        # level -> (lo, hi), log scale
    n: int = 0
    exposed: int = 0
    events_exposed: int = 0
    events_other: int = 0
    converged: bool = True

    @property
    def odds_ratio(self):
        return float(np.exp(self.log_or))

    def profile_or(self, level=0.95):
        return tuple(float(np.exp(v)) for v in self.profile[level])

    def wald_or(self, level=0.95):
        return tuple(float(np.exp(v)) for v in self.wald[level])


def g1(y, e, lps, profile_levels=(0.95,), wald_levels=(0.95,), maxit=DEFAULT_MAXIT, tol=1e-9) -> G1Result:
    """G1: Firth logistic ``Y ~ E + s(logit PS)``.

    Parameters
    ----------
    y, e : 0/1 outcome and exposure.
    lps : logit propensity score of the exposure (from :func:`nomibench.propensity.fit_ps`).
    profile_levels : confidence levels for profile penalised-likelihood intervals.
    wald_levels : confidence levels for Wald intervals (paper 1's co-primary rule uses 0.975).
    maxit : Firth iteration cap; ``REGISTERED_MAXIT`` (100) reproduces paper 1.
    """
    y = np.asarray(y, float)
    e = np.asarray(e, float)
    X = g1_design(lps, e)
    f = firth_fit(X, y, maxit=maxit, tol=tol)
    b, lp = f.beta, f.loglik
    conv = f.converged
    prof = {}
    for lev in profile_levels:
        ci = firth_profile_ci(X, y, 1, b, lp, level=lev, maxit=maxit, tol=tol)
        prof[lev] = (ci.lo, ci.hi)
        conv = conv and ci.converged
    t = firth_lrt(X, y, 1, maxit=maxit, tol=tol, full=f)
    conv = conv and t.converged
    se = wald_se(X, b, 1)
    wd = {lev: (b[1] - norm.ppf(0.5 + lev / 2) * se, b[1] + norm.ppf(0.5 + lev / 2) * se) for lev in wald_levels}
    return G1Result(float(b[1]), se, float(t.pvalue), prof, wd, len(y), int(e.sum()), int(y[e == 1].sum()),
                    int(y[e == 0].sum()), bool(conv))


def g1b(y, e, ps_quintile):
    """G1b: Mantel-Haenszel OR across propensity-score quintiles. Returns ``(OR, lo95, hi95, P)``."""
    return mh_rbg(*strat_counts(y, e, ps_quintile))


@dataclass
class Tuple4:
    """A 4-value result that unpacks as ``(estimate, lo or P, hi or O, P or E)`` and carries ``converged``."""
    values: tuple
    converged: bool = True

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, k):
        return self.values[k]


def g1c(y, e, X_oe, maxit=DEFAULT_MAXIT):
    """G1c: observed / expected events among the exposed.

    The expectation comes from a Firth logistic model ``Y ~ X_oe`` fitted on the unexposed genes only (paper 1:
    ``X_oe = [1, Tier P]``, linear). Unpacks as ``(O/E, P_two_sided, O, E)``; P is exact Poisson-binomial.
    """
    y = np.asarray(y, float)
    e = np.asarray(e, float)
    X_oe = np.asarray(X_oe, float)
    f = firth_fit(X_oe[e == 0], y[e == 0], maxit=maxit)
    pA = expit(X_oe[e == 1] @ f.beta)
    n_obs, n_exp = int(y[e == 1].sum()), pA.sum()
    p2, _ = poibin_two_sided(pA, n_obs)
    return Tuple4((n_obs / n_exp, p2, n_obs, float(n_exp)), f.converged)


def h_interaction(y, e, L, lps, maxit=DEFAULT_MAXIT):
    """H: exposure x in-locus interaction in the G1 model. Unpacks as ``(OR, lo95, hi95, P)`` for the interaction."""
    X = h_design(lps, e, L)
    t = firth_lrt(X, y, 3, maxit=maxit)
    ci = firth_profile_ci(X, y, 3, maxit=maxit)
    return Tuple4((float(np.exp(t.beta[3])), float(np.exp(ci.lo)), float(np.exp(ci.hi)), float(t.pvalue)),
                  bool(t.converged and ci.converged))


@dataclass
class W1Result:
    log_or: float
    lo: float
    hi: float
    pvalue: float
    informative_strata: int
    converged: bool

    @property
    def odds_ratio(self):
        return float(np.exp(self.log_or))


def w1(y, e, stratum, covariates=None, maxit=200, profile=True):
    """W1: conditional Firth logistic regression of Y on the exposure (plus ``covariates``) within strata.

    Paper 1: genes inside GWAS locus intervals, strata = intervals, covariates = [log10(distance to the index
    variant + 1 kb), logit PS]. Returns ``None`` when no stratum is informative.
    """
    Xw = np.asarray(e, float)[:, None] if covariates is None else np.c_[np.asarray(e, float), covariates]
    st = make_strata(Xw, y, stratum)
    if len(st) == 0:
        return None
    p = Xw.shape[1]
    f = cfl_fit(st, p, maxit=maxit)
    b, lp = f.beta, f.loglik
    b0 = b.copy()
    b0[0] = 0
    f0 = cfl_fit(st, p, fixed={0: 0.0}, beta0=b0, maxit=maxit)
    lo, hi = cfl_profile_ci(st, p, 0, b, lp, maxit=maxit) if profile else (np.nan, np.nan)
    return W1Result(float(b[0]), lo, hi, float(chi2.sf(max(2 * (lp - f0.loglik), 0), 1)), len(st),
                    bool(f.converged and f0.converged))


def w2(y, e, stratum):
    """W2: Mantel-Haenszel OR by stratum (locus interval). Returns ``(OR, lo95, hi95, P)``."""
    return mh_rbg(*strat_counts(y, e, stratum))


def naive(y, e):
    """Unadjusted Fisher exact test: ``(conditional MLE OR, exact lo95, hi95, P)``."""
    return fisher(y, e)


def unadjusted_firth(y, e, maxit=DEFAULT_MAXIT):
    """Firth logistic ``Y ~ E`` without adjustment: ``(OR, profile lo95, hi95, penalised-LRT P)``."""
    y = np.asarray(y, float)
    X = np.c_[np.ones(len(y)), np.asarray(e, float)]
    f = firth_fit(X, y, maxit=maxit)
    ci = firth_profile_ci(X, y, 1, f.beta, f.loglik, maxit=maxit)
    t = firth_lrt(X, y, 1, maxit=maxit, full=f)
    return Tuple4((float(np.exp(f.beta[1])), float(np.exp(ci.lo)), float(np.exp(ci.hi)), float(t.pvalue)),
                  bool(f.converged and ci.converged and t.converged))
