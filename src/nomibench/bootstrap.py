"""Genomic-block bootstrap of G1 (paper 1 amendment A10.3).

Neighbouring genes share GWAS signals and LD, so genes are not independent units. Genes are grouped into
contiguous genomic windows by their TSS (block = chromosome and floor(TSS / block size)); blocks are resampled
with replacement, keeping the observed number of blocks; in each replicate the propensity score and G1 are refitted.
Intervals are percentile 95%.

Arithmetic and draw order as registered (``25a_a10_hardening.py`` part 3).

Reproducibility: a resample repeats genes, so the logit propensity scores contain exact ties, and G1's spline
knots are placed on the unique values. BLAS rounding can split a tie by one ulp on a different platform, which
moves a knot. Replicates are exactly reproducible within an environment, but agree only to about 1e-4 across
environments. Percentile intervals over many replicates are unaffected in practice.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from ._util import check_table
from .estimators import g1_design
from .firth import DEFAULT_MAXIT, firth_fit
from .propensity import fit_ps

__all__ = ["make_blocks", "run", "summarise"]


def make_blocks(chrom, tss, block_size):
    """Gene indices per block, blocks numbered in order of first appearance."""
    win = pd.Series(np.floor(np.asarray(tss, float) / block_size)).astype(int).astype(str)
    key = pd.Series(chrom).astype(str).reset_index(drop=True) + ":" + win
    codes, _ = pd.factorize(key)
    return [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]


_G = {}


def _init(bundle):
    _G.clear()
    _G.update(bundle)


def _boot(args):
    k, seed = args
    rng = np.random.default_rng(seed)
    blocks = _G["blocks"]
    pick = rng.integers(0, len(blocks), len(blocks))
    idx = np.concatenate([blocks[i] for i in pick])
    Db = _G["D"].iloc[idx].reset_index(drop=True)
    yb = _G["y"][idx]
    res = [k, len(idx), float(yb.sum())]
    for ex in _G["exposures"]:
        try:
            Dx = Db.assign(**{_G["col"]: Db[ex].astype(float).values})
            pss = fit_ps(Dx, _G["col"], _G["cols"], _G["linear"])
            b = firth_fit(g1_design(pss.logit, Dx[_G["col"]].values), yb, maxit=_G["maxit"], warn=False).beta
            res.append(float(b[1]))
        except Exception:
            res.append(np.nan)
    return tuple(res)


def run(df: pd.DataFrame, outcome: str, exposures, cols, linear=(), chrom="chr", tss="tss", block_size=5e6,
        n_reps=2000, seed=None, seeds=None, workers=1, maxit=DEFAULT_MAXIT, exposure_column="E") -> pd.DataFrame:
    """Block bootstrap of the G1 log odds ratio for each exposure in ``exposures``.

    One child of ``SeedSequence(seed)`` per replicate, or explicit ``seeds``. Returns one row per replicate:
    ``rep, n_genes, events, lnOR_<exposure>...``. A replicate whose fit fails gets NaN for that exposure.
    """
    check_table(df, outcome, list(exposures), cols)
    if df[chrom].isna().any() or not np.all(np.isfinite(pd.to_numeric(df[tss], errors="coerce").values)):
        raise ValueError(f"{chrom!r} and {tss!r} must be complete (TSS numeric)")
    blocks = make_blocks(df[chrom].values, df[tss].values, block_size)
    if seeds is None:
        if seed is None:
            raise ValueError("a seed is required, so that the bootstrap can be reproduced")
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        ss = np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key, pool_size=ss.pool_size)   # reuse-safe copy
        seeds = ss.spawn(n_reps)
    jobs = list(enumerate(seeds))
    bundle = dict(D=df, y=df[outcome].values.astype(float), blocks=blocks, exposures=tuple(exposures),
                  cols=tuple(cols), linear=tuple(linear), maxit=maxit, col=exposure_column)
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(bundle,)) as pool:
            r = pool.map(_boot, jobs, chunksize=8)
    else:
        _init(bundle)
        r = [_boot(j) for j in jobs]
    return pd.DataFrame(r, columns=["rep", "n_genes", "events"] + [f"lnOR_{e}" for e in exposures])


def summarise(B: pd.DataFrame, exposure, level=0.95):
    """Bootstrap SE of log OR and the percentile interval (OR scale); failed replicates are dropped and counted."""
    v = B[f"lnOR_{exposure}"].dropna().values
    a = round((1 - level) / 2, 12)              # exactly 0.025 at level 0.95, as registered
    lo, hi = np.quantile(v, [a, round(1 - a, 12)])
    return dict(exposure=exposure, reps=len(B), failed=int(B[f"lnOR_{exposure}"].isna().sum()),
                boot_se=float(v.std(ddof=1)), pct_lo=float(np.exp(lo)), pct_hi=float(np.exp(hi)))
