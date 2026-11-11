"""Discoverability-matched negative controls on the real outcome.

Random gene sets the size of the exposure are drawn, without replacement, from the UNEXPOSED genes with
probability proportional to the exposure's propensity odds ps / (1 - ps), so that each set resembles the exposure
in discoverability but carries no nomination. Each set gets its own propensity score and G1 fit on the real
outcomes of the unexposed genes. Under a calibrated pipeline about 5% of sets reject at P < 0.05; paper 1's
registered pass band is 3.5-6.5% (spec v2 section 3).

Arithmetic and draw order as registered (``19b_benchmark.py`` E4; ``20b_e5_calibration.py``; ``25a`` A10.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import kstest

from ._util import check_table
from .estimators import g1_design
from .firth import DEFAULT_MAXIT, firth_lrt
from .propensity import fit_ps

__all__ = ["draw_sets", "run", "NCSummary", "summarise"]


def draw_sets(ps, eligible, size, n_sets, rng):
    """Index sets (into the eligible genes, in their order) drawn without replacement with probability
    proportional to the propensity odds. ``rng`` is a ``numpy.random.Generator``; sets are drawn in sequence."""
    ps = np.asarray(ps, float)[np.asarray(eligible, bool)]
    odds = ps / (1 - ps)
    return [rng.choice(len(ps), size, replace=False, p=odds / odds.sum()) for _ in range(n_sets)]


_G = {}


def _init(bundle):
    _G.clear()
    _G.update(bundle)


def _one(args):
    k, idx = args
    D0, y0 = _G["D0"], _G["y0"]
    S = np.zeros(len(D0))
    S[idx] = 1.0
    try:
        pss = fit_ps(D0.assign(**{_G["col"]: S}), _G["col"], _G["cols"], _G["linear"])
        t = firth_lrt(g1_design(pss.logit, S), y0, 1, maxit=_G["maxit"])
        return k, float(t.beta[1]), float(t.pvalue), float(y0[idx].sum())
    except Exception:
        return k, np.nan, np.nan, float(y0[idx].sum())


def run(df: pd.DataFrame, outcome: str, exposure: str, cols, linear=(), n_sets=1000, seed=None, rng=None,
        ps=None, workers=1, maxit=DEFAULT_MAXIT, set_column="S") -> pd.DataFrame:
    """Matched negative control for ``exposure``.

    The exposure's propensity score is fitted on ``df`` (or passed as ``ps``); sets of the exposure's size are
    drawn from the unexposed genes; each set's G1 uses the unexposed genes' outcomes only, so no exposed gene's
    outcome is read. Pass ``rng`` (a Generator) or ``seed``; one is required. Returns one row per set:
    ``set, logor, p, events`` (a set whose fit fails gets NaN, counted as a non-rejection).
    """
    check_table(df, outcome, exposure, cols, outcome_complete=False)
    if rng is None and seed is None:
        raise ValueError("give `seed` or `rng`, so that the sets can be reproduced")
    E = df[exposure].values.astype(float)
    if ps is None:
        ps = fit_ps(df, exposure, cols, linear).ps
    rng = rng if rng is not None else np.random.default_rng(seed)
    nonE = E == 0
    D0 = df[nonE].reset_index(drop=True)
    y0 = df[outcome].values.astype(float)[nonE]
    sets = list(enumerate(draw_sets(ps, nonE, int(E.sum()), n_sets, rng)))
    bundle = dict(D0=D0, y0=y0, cols=tuple(cols), linear=tuple(linear), maxit=maxit, col=set_column)
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(bundle,)) as pool:
            r = pool.map(_one, sets, chunksize=10)
    else:
        _init(bundle)
        r = [_one(s) for s in sets]
    return pd.DataFrame(r, columns=["set", "logor", "p", "events"])


@dataclass
class NCSummary:
    sets: int
    failed: int
    reject: float
    mc_se: float
    ks_p: float
    or_median: float
    or_q025: float
    or_q975: float
    passes: bool


def summarise(R: pd.DataFrame, alpha=0.05, band=(0.035, 0.065)) -> NCSummary:
    """Rejection rate at P < alpha (failed fits count as non-rejections), Monte Carlo SE, KS test of the set
    P-values against U(0, 1), OR quantiles, and the band verdict."""
    pv = R.p.values.astype(float)
    rej = float(np.mean(np.nan_to_num(pv, nan=1.0) < alpha))
    se = float(np.sqrt(rej * (1 - rej) / len(R)))
    ks = float(kstest(pv[np.isfinite(pv)], "uniform").pvalue)
    q = np.exp(np.nanquantile(R.logor.values.astype(float), [0.025, 0.5, 0.975]))
    return NCSummary(len(R), int(np.isnan(pv).sum()), rej, se, ks, float(q[1]), float(q[0]), float(q[2]),
                     bool(band[0] <= rej <= band[1]))
