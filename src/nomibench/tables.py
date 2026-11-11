"""2x2 and stratified-table methods, and the exact Poisson-binomial tail.

  * Fisher's exact test with the conditional maximum-likelihood odds ratio and its exact CI (scipy);
  * Mantel-Haenszel common odds ratio with the Robins-Breslow-Greenland variance (Mantel & Haenszel 1959;
    Robins, Breslow & Greenland 1986, Biometrics 42:311, PMID 3741973);
  * exact Poisson-binomial distribution by dynamic programming, for observed vs expected counts.

Arithmetic as registered (``mbstats.py``, ``gwcore.py``, ``19b_benchmark.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import fisher_exact, norm
from scipy.stats.contingency import odds_ratio

__all__ = ["FisherResult", "fisher", "fisher_logor", "strat_counts", "mh_rbg", "poibin_pmf", "poibin_two_sided"]


def _table(y, a):
    y = np.asarray(y, float)
    a = np.asarray(a, float)
    return [[int(y[a == 1].sum()), int((1 - y[a == 1]).sum())], [int(y[a == 0].sum()), int((1 - y[a == 0]).sum())]]


@dataclass
class FisherResult:
    """Conditional MLE odds ratio, its exact 95% CI and the two-sided Fisher P. Unpacks as ``OR, lo, hi, P``."""
    odds_ratio: float
    lo: float
    hi: float
    pvalue: float

    def __iter__(self):
        yield self.odds_ratio
        yield self.lo
        yield self.hi
        yield self.pvalue


def fisher(y, a, level=0.95) -> FisherResult:
    """Unadjusted comparison of outcome ``y`` between exposed (``a`` = 1) and other units."""
    t = _table(y, a)
    r = odds_ratio(t, kind="conditional")
    ci = r.confidence_interval(level)
    return FisherResult(r.statistic, ci.low, ci.high, fisher_exact(t)[1])


def fisher_logor(y, a):
    """(log sample odds ratio, two-sided Fisher P), the per-replicate form used in the plasmode.
    The log OR is NaN when the sample odds ratio is 0 or infinite."""
    y = np.asarray(y, float)
    a = np.asarray(a, float)
    tab = [[int((y * a).sum()), int(((1 - y) * a).sum())], [int((y * (1 - a)).sum()), int(((1 - y) * (1 - a)).sum())]]
    o, p = fisher_exact(tab)
    return (np.log(o) if 0 < o < np.inf else np.nan), p


def strat_counts(y, a, strata):
    """Per-stratum counts (exposed events, exposed non-events, unexposed events, unexposed non-events)."""
    y = np.asarray(y, float)
    a = np.asarray(a, float)
    _, k = np.unique(strata, return_inverse=True)
    K = k.max() + 1

    def f(w):
        return np.bincount(k, weights=w, minlength=K)

    return f(y * a), f((1 - y) * a), f(y * (1 - a)), f((1 - y) * (1 - a))


def mh_rbg(a, b, c, d):
    """Mantel-Haenszel common OR over strata with the RBG variance.

    a = exposed events, b = exposed non-events, c = unexposed events, d = unexposed non-events (arrays over
    strata). Returns ``(OR, lo95, hi95, P)``; P is the Wald test of log OR = 0. All NaN if undefined.
    """
    a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))
    n = a + b + c + d
    keep = n > 0
    a, b, c, d, n = a[keep], b[keep], c[keep], d[keep], n[keep]
    R = a * d / n
    S = b * c / n
    if R.sum() == 0 or S.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    orr = R.sum() / S.sum()
    P = (a + d) / n
    Q = (b + c) / n
    var = (np.sum(P * R) / (2 * R.sum() ** 2)
           + np.sum(P * S + Q * R) / (2 * R.sum() * S.sum())
           + np.sum(Q * S) / (2 * S.sum() ** 2))
    se = np.sqrt(var)
    lo = np.log(orr)
    z = norm.ppf(0.975)
    return orr, np.exp(lo - z * se), np.exp(lo + z * se), 2 * norm.sf(abs(lo) / se)


def poibin_pmf(p):
    """Exact distribution of a sum of independent Bernoulli(p_i) variables."""
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for q in p:
        pmf[1:] = pmf[1:] * (1 - q) + pmf[:-1] * q
        pmf[0] *= (1 - q)
    return pmf


def poibin_two_sided(p, obs):
    """Exact two-sided P for an observed count ``obs`` (sum of the probabilities of outcomes no more probable than
    the observed one), and the upper tail P(X >= obs)."""
    pmf = poibin_pmf(np.asarray(p, float))
    return float(min(1.0, pmf[pmf <= pmf[obs] * (1 + 1e-9)].sum())), float(pmf[obs:].sum())
