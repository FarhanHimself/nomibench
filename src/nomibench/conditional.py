"""Exact conditional (stratified) logistic regression with Firth's penalty.

Methods:
  * conditional likelihood for matched or stratified binary data (Breslow & Day 1980, ch. 7);
  * Firth-penalised conditional likelihood, "conditional Firth logistic regression"
    (Heinze & Puhr 2010, Stat Med 29:770, doi:10.1002/sim.3794).

Two exact evaluators of the same conditional likelihood:
  * enumeration of every size-m subset of a stratum, used when a stratum has at most :data:`ENUM_MAX` subsets;
  * a dynamic programme over the units of a stratum: the recursion of Gail, Lubin & Rubinstein (1981,
    Biometrika 68:703, doi:10.1093/biomet/68.3.703) for the conditional-Bernoulli normaliser, extended to carry
    the mean, covariance and third central moment of the subset-sum statistic. Exact, O(n m p^3), no size limit.

The penalty gradient is analytic: d/dbeta_k 1/2 log|I| = 1/2 tr(I^-1 dI/dbeta_k), and dI/dbeta_k is the third
central moment of the subset-sum statistic.

Arithmetic as registered (``mbstats.py`` with amendment A4); the package adds a convergence record.
Validated in paper 1 (A10.8) against R ``survival::clogit(method = "exact")``: coefficients within 2.3e-15.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

from ._util import warn_not_converged

ENUM_MAX = 20_000
_COMB = {}

__all__ = ["Strata", "make_strata", "cfl_fit", "cfl_lrt", "cfl_profile_ci", "conditional_loglik", "CFLFit",
           "CFLTest", "ENUM_MAX"]


def _comb_idx(n, m):
    key = (n, m)
    if key not in _COMB:
        _COMB[key] = np.array(list(combinations(range(n), m)), int).reshape(-1, m)
    return _COMB[key]


def _prep_stratum(Xs, ys):
    """Precompute the beta-independent parts of one stratum. If more than half the stratum are events,
    condition on the non-events instead (covariates negated): identical likelihood, score, information and
    third moment, fewer subsets."""
    n = len(ys)
    m = int(ys.sum())
    sgn = 1.0
    if m > n - m:
        ys = 1 - ys
        m = n - m
        sgn = -1.0
    Xs2 = sgn * np.asarray(Xs, float)
    Tobs = Xs2[np.asarray(ys) == 1].sum(axis=0)
    if comb(n, m) <= ENUM_MAX:
        return ("E", Xs2[_comb_idx(n, m)].sum(axis=1), Tobs)      # (nsub, p) subset sums
    return ("D", Xs2, m, Tobs)


class Strata(list):
    """Informative strata (at least one event and one non-event); ``len()`` is their number.
    Strata too large to enumerate are also packed into one zero-padded batch for the dynamic programme."""

    def __init__(self, items):
        super().__init__(items)
        big = [s for s in items if s[0] == "D"]
        self.dp = None
        if big:
            S = len(big)
            nmax = max(len(s[1]) for s in big)
            p = big[0][1].shape[1]
            Xp = np.zeros((S, nmax, p))
            valid = np.zeros((S, nmax), bool)
            for i, s in enumerate(big):
                Xp[i, :len(s[1])] = s[1]
                valid[i, :len(s[1])] = True
            self.dp = (Xp, valid, np.array([s[2] for s in big], int), np.array([s[3] for s in big]))


def make_strata(X, y, sid) -> Strata:
    """Informative strata only, precomputed. ``sid`` labels each unit's stratum (e.g. a GWAS locus interval)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    sid = np.asarray(sid)
    out = []
    for s in np.unique(sid):
        k = sid == s
        if 0 < y[k].sum() < k.sum():
            out.append(_prep_stratum(X[k], y[k]))
    return Strata(out)


def _sym3(C, d):
    """C_ab d_c + C_ac d_b + d_a C_bc, over any leading batch shape."""
    return (C[..., :, :, None] * d[..., None, None, :] + C[..., :, None, :] * d[..., None, :, None]
            + d[..., :, None, None] * C[..., None, :, :])


def _cube(d):
    return d[..., :, None, None] * d[..., None, :, None] * d[..., None, None, :]


def _dp_moments(Xp, valid, m, beta, third):
    """For a padded batch of strata: log normaliser, mean, covariance and third central moment of
    T = sum_{i in S} x_i under P(S) proportional to exp(beta'T), |S| = m. State (stratum, size j) after units
    1..k is a distribution over subsets; adding unit k+1 mixes 'exclude' (size j) with 'include' (size j-1,
    shifted by x_{k+1}) in the proportion of their normalisers, kept on the log scale. Padded units never enter."""
    S, nmax, p = Xp.shape
    J = int(m.max())
    lr = np.where(valid, Xp @ beta, -np.inf)
    lW = np.full((S, J + 1), -np.inf)
    lW[:, 0] = 0.0
    mu = np.zeros((S, J + 1, p))
    C = np.zeros((S, J + 1, p, p))
    K = np.zeros((S, J + 1, p, p, p)) if third else None
    with np.errstate(invalid="ignore", over="ignore"):
        for k in range(nmax):
            la = lr[:, k, None] + lW[:, :-1]
            lb = lW[:, 1:]
            ln = np.logaddexp(la, lb)
            a = np.exp(la - ln)
            a = np.where(np.isfinite(ln) & np.isfinite(a), a, 0.0)
            a1 = a[..., None]
            a2 = a1[..., None]
            ma = mu[:, :-1] + Xp[:, k, None, :]
            mb = mu[:, 1:]
            mn = (1 - a1) * mb + a1 * ma
            da = ma - mn
            db = mb - mn
            Ca, Cb = C[:, :-1], C[:, 1:]
            Cn = (1 - a2) * (Cb + db[..., :, None] * db[..., None, :]) + a2 * (Ca + da[..., :, None] * da[..., None, :])
            if third:
                a3 = a2[..., None]
                K[:, 1:] = ((1 - a3) * (K[:, 1:] + _sym3(Cb, db) + _cube(db))
                            + a3 * (K[:, :-1] + _sym3(Ca, da) + _cube(da)))
            C[:, 1:] = Cn
            mu[:, 1:] = mn
            lW[:, 1:] = ln
    r = np.arange(S)
    return lW[r, m], mu[r, m], C[r, m], (K[r, m] if third else None)


def _cond_all(strata, beta, third=False):
    """Exact conditional log-likelihood, score, information and (optionally) the third central moment tensor,
    summed over strata."""
    p = len(beta)
    ll = 0.0
    U = np.zeros(p)
    I = np.zeros((p, p))
    M3 = np.zeros((p, p, p)) if third else None
    for s in strata:
        if s[0] != "E":
            continue
        _, T, Tobs = s
        e = T @ beta
        emax = e.max()
        w = np.exp(e - emax)
        Z = w.sum()
        w /= Z
        mu = w @ T
        D = T - mu
        I += (D * w[:, None]).T @ D
        U += Tobs - mu
        ll += Tobs @ beta - (emax + np.log(Z))
        if third:                                    # sum_s w_s D_s (x) D_s (x) D_s as one BLAS product
            M3 += ((D * w[:, None]).T @ (D[:, :, None] * D[:, None, :]).reshape(len(D), -1)).reshape(p, p, p)
    dp = getattr(strata, "dp", None)
    if dp is not None:
        Xp, valid, m, Tobs = dp
        lZ, mu, Cv, K3 = _dp_moments(Xp, valid, m, beta, third)
        ll += float((Tobs @ beta).sum() - lZ.sum())
        U += (Tobs - mu).sum(axis=0)
        I += Cv.sum(axis=0)
        if third:
            M3 += K3.sum(axis=0)
    return ll, U, I, M3


def conditional_loglik(strata, beta, penalised=True):
    """Conditional log-likelihood at ``beta``, plus 1/2 log|I| when ``penalised``."""
    if penalised:
        return _cfl_pen(strata, np.asarray(beta, float))
    return _cond_all(strata, np.asarray(beta, float))[0]


def _cfl_pen(strata, beta):
    ll, U, I, _ = _cond_all(strata, beta)
    sign, logdet = np.linalg.slogdet(I)
    return (ll + 0.5 * logdet if sign > 0 else -np.inf)


@dataclass
class CFLFit:
    """Result of :func:`cfl_fit`; unpacks as ``beta, loglik``."""
    beta: np.ndarray
    loglik: float
    converged: bool
    n_iter: int

    def __iter__(self):
        yield self.beta
        yield self.loglik


def cfl_fit(strata, p, fixed=None, beta0=None, maxit=200, tol=1e-8, penalised=True, warn=True) -> CFLFit:
    """Maximise the (Firth-penalised, by default) conditional log-likelihood by Newton steps with the
    conditional information, step capped at 3 and halved until the objective does not decrease."""
    fixed = fixed or {}
    free = np.array([j for j in range(p) if j not in fixed], int)
    beta = np.zeros(p) if beta0 is None else np.array(beta0, float)
    for j, v in fixed.items():
        beta[j] = v

    def obj(bb):
        return _cfl_pen(strata, bb) if penalised else _cond_all(strata, bb)[0]

    cur = obj(beta)
    if len(free) == 0:
        return CFLFit(beta, cur, True, 0)
    converged, it = False, -1
    for it in range(maxit):
        _, U, I, M3 = _cond_all(strata, beta, third=penalised)
        g = U.copy()
        if penalised:
            Iinv = np.linalg.pinv(I)
            g += 0.5 * np.einsum("ij,ijk->k", Iinv, M3)
        gf = g[free]
        If = I[np.ix_(free, free)] + 1e-10 * np.eye(len(free))
        step = np.linalg.solve(If, gf)
        mx = np.max(np.abs(step))
        if mx > 3:
            step *= 3 / mx
        t = 1.0
        while True:
            nb = beta.copy()
            nb[free] += t * step
            nv = obj(nb)
            if nv >= cur - 1e-12 or t < 1e-6:
                break
            t /= 2
        done = np.max(np.abs(nb - beta)) < tol
        beta, cur = nb, nv
        if done:
            converged = True
            break
    if not converged and warn:
        warn_not_converged("cfl_fit", maxit)
    return CFLFit(beta, cur, converged, it + 1)


@dataclass
class CFLTest:
    """Penalised conditional likelihood-ratio test; unpacks as ``beta, stat, pvalue, loglik``."""
    beta: np.ndarray
    stat: float
    pvalue: float
    loglik: float
    converged: bool

    def __iter__(self):
        yield self.beta
        yield self.stat
        yield self.pvalue
        yield self.loglik


def cfl_lrt(strata, p, j, maxit=200, tol=1e-8) -> CFLTest:
    f = cfl_fit(strata, p, maxit=maxit, tol=tol)
    b, lp = f.beta, f.loglik
    b0 = b.copy()
    b0[j] = 0.0
    f0 = cfl_fit(strata, p, fixed={j: 0.0}, beta0=b0, maxit=maxit, tol=tol)
    stat = max(2 * (lp - f0.loglik), 0.0)
    return CFLTest(b, stat, chi2.sf(stat, 1), lp, bool(f.converged and f0.converged))


def cfl_profile_ci(strata, p, j, b, lp, level=0.95, span=10.0, maxit=200, tol=1e-8):
    """Profile penalised conditional-likelihood CI for beta_j (Brent, absolute tolerance 1e-5). Returns
    ``(lo, hi)``; NaN where a bound is not bracketed within ``span``."""
    crit = chi2.ppf(level, 1) / 2

    def g(v):
        _, lpv = cfl_fit(strata, p, fixed={j: v}, beta0=b, maxit=maxit, tol=tol, warn=False)
        return (lp - lpv) - crit

    lo = hi = np.nan
    try:
        lo = brentq(g, b[j] - span, b[j], xtol=1e-5)
    except ValueError:
        pass
    try:
        hi = brentq(g, b[j], b[j] + span, xtol=1e-5)
    except ValueError:
        pass
    return lo, hi
