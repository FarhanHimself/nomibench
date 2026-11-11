"""Firth-penalised logistic regression.

Methods:
  * penalised maximum likelihood with Jeffreys' invariant prior (Firth 1993, Biometrika 80:27);
  * penalised likelihood-ratio test and profile penalised-likelihood confidence interval
    (Heinze & Schemper 2002, Stat Med 21:2409, doi:10.1002/sim.1047).

When a coefficient is held fixed (restricted fits for the test and the profile), the penalty is computed from the
FULL design, as in R's ``logistf``.

The arithmetic is the registered code's (``mbstats.py``, paper 1 steps 14-27), unchanged. The package adds a
convergence record and a warning when the iteration cap binds. The registered analyses used ``maxit=100``
(:data:`REGISTERED_MAXIT`); the package default is 1,000. Of paper 1's ten registered G1 fits, only the E1 fit
stopped at the cap, and its log odds ratio moves by 3.4e-8 at the default. Replicate, negative-control and
generator fits were not re-examined at the higher cap; passing ``maxit=REGISTERED_MAXIT`` reproduces them exactly.

``converged`` is the registered stopping rule: both the largest coefficient change and the change in the penalised
log-likelihood fell below ``tol``. Along a nearly flat direction this can hold while the score is still about
1e-7, so ``max_abs_score`` is reported as well.

Fitted probabilities are clipped to [1e-15, 1 - 1e-15] inside the likelihood, as registered. Under (quasi-)complete
separation a profile bound can lie where the clip binds; :func:`firth_profile_ci` then flags the interval as
``saturated``, and that bound should be read as unbounded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from ._util import expit, warn_not_converged

DEFAULT_MAXIT = 1000
REGISTERED_MAXIT = 100

__all__ = ["FirthFit", "LRTResult", "ProfileCI", "firth_fit", "firth_lrt", "firth_profile_ci", "wald_se",
           "wald_ci", "DEFAULT_MAXIT", "REGISTERED_MAXIT"]


def _pen_loglik(X, y, beta):
    eta = X @ beta
    pi = 1.0 / (1.0 + np.exp(-eta))
    pi = np.clip(pi, 1e-15, 1 - 1e-15)
    ll = np.sum(y * np.log(pi) + (1 - y) * np.log1p(-pi))
    W = pi * (1 - pi)
    XtWX = X.T @ (X * W[:, None])
    sign, logdet = np.linalg.slogdet(XtWX)
    if sign <= 0:
        return -np.inf, pi, W, XtWX
    return ll + 0.5 * logdet, pi, W, XtWX


def _modified_score(X, y, pi, W, XtWX):
    try:
        inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(XtWX)
    h = W * np.einsum("ij,jk,ik->i", X, inv, X)
    return X.T @ (y - pi + h * (0.5 - pi))


@dataclass
class FirthFit:
    """Result of :func:`firth_fit`.

    Unpacks as ``beta, loglik = firth_fit(...)`` for compatibility with the registered scripts.

    Attributes
    ----------
    beta : coefficients (fixed ones included).
    loglik : penalised log-likelihood at ``beta``.
    converged : whether the convergence criterion was met before the iteration cap.
    n_iter : iterations used.
    max_abs_score : largest absolute modified score over the free coefficients at ``beta`` (0 at the optimum).
    """
    beta: np.ndarray
    loglik: float
    converged: bool
    n_iter: int
    max_abs_score: float

    def __iter__(self):
        yield self.beta
        yield self.loglik


def firth_fit(X, y, fixed=None, beta0=None, maxit=DEFAULT_MAXIT, tol=1e-9, warn=True) -> FirthFit:
    """Penalised maximum likelihood for logistic regression (Firth 1993).

    Newton iterations with the expected information, step capped at 5 and halved until the penalised
    log-likelihood does not decrease. Converged when both the largest coefficient change and the change in the
    penalised log-likelihood are below ``tol``.

    Parameters
    ----------
    X : (n, p) design, intercept included.
    y : (n,) 0/1 outcome.
    fixed : {column index: value} held fixed (restricted fit; the penalty still uses the full design).
    beta0 : starting values.
    maxit : iteration cap. Use :data:`REGISTERED_MAXIT` to reproduce paper 1.
    tol : convergence tolerance.
    warn : emit a :class:`~nomibench.ConvergenceWarning` if the cap binds.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    fixed = fixed or {}
    free = np.array([j for j in range(p) if j not in fixed], int)
    beta = np.zeros(p) if beta0 is None else np.array(beta0, float)
    for j, v in fixed.items():
        beta[j] = v
    lp, pi, W, XtWX = _pen_loglik(X, y, beta)
    if len(free) == 0:
        return FirthFit(beta, lp, True, 0, 0.0)
    converged, it = False, -1
    for it in range(maxit):
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
        h = W * np.einsum("ij,jk,ik->i", X, XtWX_inv, X)        # leverages of the full design
        U = X.T @ (y - pi + h * (0.5 - pi))                      # Firth's modified score
        Uf = U[free]
        If = XtWX[np.ix_(free, free)]
        try:
            step = np.linalg.solve(If, Uf)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(If, Uf, rcond=None)[0]
        mx = np.max(np.abs(step))
        if mx > 5:
            step *= 5 / mx
        t = 1.0
        while True:
            nb = beta.copy()
            nb[free] += t * step
            nlp, npi, nW, nXtWX = _pen_loglik(X, y, nb)
            if nlp >= lp - 1e-12 or t < 1e-6:
                break
            t /= 2
        conv = np.max(np.abs(nb - beta)) < tol and abs(nlp - lp) < tol
        beta, lp, pi, W, XtWX = nb, nlp, npi, nW, nXtWX
        if conv:
            converged = True
            break
    score = _modified_score(X, y, pi, W, XtWX)[free]
    if not converged and warn:
        warn_not_converged("firth_fit", maxit)
    return FirthFit(beta, lp, converged, it + 1, float(np.max(np.abs(score))))


@dataclass
class LRTResult:
    """Penalised likelihood-ratio test of beta_j = 0. Unpacks as ``beta, stat, pvalue``."""
    beta: np.ndarray
    stat: float
    pvalue: float
    converged: bool

    def __iter__(self):
        yield self.beta
        yield self.stat
        yield self.pvalue

    def __getitem__(self, k):
        return (self.beta, self.stat, self.pvalue)[k]


def firth_lrt(X, y, j, maxit=DEFAULT_MAXIT, tol=1e-9, full=None) -> LRTResult:
    """Penalised likelihood-ratio test of H0: beta_j = 0 (Heinze & Schemper 2002), chi-squared on 1 df.

    ``full`` may pass an existing unrestricted :class:`FirthFit` of the same ``X``, ``y``, ``maxit`` and ``tol``;
    the fit is deterministic, so reusing it gives the identical result.
    """
    X = np.asarray(X, float)
    f = full if full is not None else firth_fit(X, y, maxit=maxit, tol=tol)
    b, lp = f.beta, f.loglik
    f0 = firth_fit(X, y, fixed={j: 0.0}, beta0=np.where(np.arange(X.shape[1]) == j, 0, b), maxit=maxit, tol=tol)
    stat = max(2 * (lp - f0.loglik), 0.0)
    return LRTResult(b, stat, chi2.sf(stat, 1), bool(f.converged and f0.converged))


@dataclass
class ProfileCI:
    """Profile penalised-likelihood confidence interval (log scale). Unpacks as ``lo, hi``; NaN if a bound
    was not bracketed within ``span``. ``saturated`` is True if, at either bound, some fitted probability reached
    the 1e-15 clip (typically under separation): that bound is then not reliable and should be read as unbounded."""
    lo: float
    hi: float
    converged: bool
    saturated: bool = False

    def __iter__(self):
        yield self.lo
        yield self.hi


def firth_profile_ci(X, y, j, b=None, lp=None, level=0.95, span=12.0, maxit=DEFAULT_MAXIT, tol=1e-9) -> ProfileCI:
    """Profile penalised-likelihood CI for beta_j: the set {v: 2 [lp - max lp(beta_j = v)] <= chi2_{1,level}}.

    Each bound is found by Brent's method (absolute tolerance 1e-6 on beta_j) within ``span`` of the estimate.
    ``converged`` is False if the restricted fit at either bound met its iteration cap.
    """
    X = np.asarray(X, float)
    if b is None:
        f = firth_fit(X, y, maxit=maxit, tol=tol)
        b, lp = f.beta, f.loglik
    crit = chi2.ppf(level, 1) / 2
    seen = []                                    # (v, converged, saturated) of every restricted fit

    def g(v):
        fv = firth_fit(X, y, fixed={j: v}, beta0=b, maxit=maxit, tol=tol, warn=False)
        p_ = expit(X @ fv.beta)
        seen.append((v, fv.converged, bool(np.any(p_ < 1e-15) or np.any(p_ > 1 - 1e-15))))
        return (lp - fv.loglik) - crit

    def at_root(r):
        # the restricted fit evaluated closest to the root; fits at the far bracket ends, where probabilities
        # saturate, may creep without affecting the root
        return min(seen, key=lambda t: abs(t[0] - r))

    lo = hi = np.nan
    ok, sat = True, False
    try:
        lo = brentq(g, b[j] - span, b[j], xtol=1e-6)
        _, c, s_ = at_root(lo)
        ok, sat = ok and c, sat or s_
    except ValueError:
        pass
    try:
        hi = brentq(g, b[j], b[j] + span, xtol=1e-6)
        _, c, s_ = at_root(hi)
        ok, sat = ok and c, sat or s_
    except ValueError:
        pass
    if not ok:
        warn_not_converged("firth_profile_ci (a restricted fit at a bound)", maxit)
    return ProfileCI(lo, hi, ok, sat)


def wald_se(X, beta, j):
    """Wald standard error of beta_j from the (unpenalised) expected information at ``beta``, as in the
    registered code: ``sqrt(inv(X' W X)[j, j])``."""
    X = np.asarray(X, float)
    pi = expit(X @ beta)
    W = pi * (1 - pi)
    return float(np.sqrt(np.linalg.inv(X.T @ (X * W[:, None]))[j, j]))


def wald_ci(estimate, se, level=0.95, z=None):
    """Two-sided Wald interval ``estimate -/+ z se``; ``z`` defaults to the normal quantile for ``level``."""
    z = norm.ppf(0.5 + level / 2) if z is None else z
    return estimate - z * se, estimate + z * se
