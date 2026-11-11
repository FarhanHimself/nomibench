"""One-call entry points for a nomination benchmark on a gene table.

The table has one row per testable gene, a 0/1 outcome (e.g. SMR/HEIDI support), a 0/1 exposure (the
nomination), discoverability covariates and, optionally, GWAS-locus columns. Typical use::

    import nomibench as nb
    res = nb.evaluate(genes, outcome="Y", exposure="E", covariates=["x_z", "x_N", "x_len"])
    cal = nb.calibrate(genes, outcome="Y", exposure="E", covariates=[...], generator_covariates=["x_z", "x_len"],
                       locus="interval", locus_distance="ld", seed=1)
    sets, nc = nb.matched_controls(genes, outcome="Y", exposure="E", covariates=[...], seed=2)

The exposure's outcomes are never read by :func:`calibrate` or :func:`matched_controls`, so both can be run, and
their verdicts fixed, before the contrast they govern.

Locus columns: ``locus`` holds a GWAS-locus interval identifier (any type) and is empty (NaN/None) for genes outside
loci; ``locus_distance`` holds log10(distance to the interval's nearest index variant + 1 kb) for in-locus genes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import negative_control as _nc
from . import plasmode as _pm
from ._util import check_table
from .decisions import NC_BAND, VALIDITY_THRESHOLD
from .estimators import g1, g1_design, g1b, g1c, h_design, h_interaction, naive, w1
from .firth import DEFAULT_MAXIT, firth_fit
from .propensity import balance, fit_ps

__all__ = ["evaluate", "calibrate", "matched_controls", "Calibration", "check_table", "locus_arrays"]


def locus_arrays(df, locus, locus_distance=None):
    """``(in_locus, locus_id, ld)`` from the locus columns: ``locus_id`` is 0..K-1 inside loci and -1 outside
    (identifiers are factorised in sorted order, so any type works and integer ids 0..K-1 keep their values);
    ``ld`` is the distance term, 0 outside loci, or None if no distance column is given."""
    if locus is None:
        return None, None, None
    lid = df[locus]
    inl = lid.notna().values
    if inl.mean() > 0.5:
        warnings.warn(f"{inl.mean():.0%} of genes have a {locus!r} value; genes outside loci must be empty (NaN), "
                      "not coded 0", UserWarning, stacklevel=3)
    iid = np.full(len(df), -1, dtype=int)
    iid[inl] = pd.factorize(lid[inl], sort=True)[0]       # sorted: integer ids 0..K-1 map to themselves
    ld = None
    if locus_distance is not None:
        v = df[locus_distance].values.astype(float)
        if np.isnan(v[inl]).any():
            raise ValueError(f"{locus_distance!r} is missing for some in-locus genes")
        ld = np.where(inl, v, 0.0)
    return inl, iid, ld


def _w1_identified(y, e, sid):
    """The within-locus exposure effect is identified only if some locus holds both events and non-events (an
    informative stratum) and both exposed and unexposed genes."""
    for s in np.unique(sid):
        k = sid == s
        if 0 < y[k].sum() < k.sum() and 0 < e[k].sum() < k.sum():
            return True
    return False


def evaluate(df: pd.DataFrame, outcome: str, exposure: str, covariates, linear=(), locus=None, locus_distance=None,
             maxit=DEFAULT_MAXIT, wald_levels=(0.95, 0.975), profile_levels=(0.95,)) -> pd.DataFrame:
    """Estimate enrichment of ``outcome`` among ``exposure`` genes, adjusted for discoverability.

    Rows: naive Fisher; G1 (primary; profile and Wald intervals); G1b; G1c; and, when ``locus`` is given, the
    exposure x in-locus interaction (H) and the within-locus conditional estimate (W1; adjusted for the distance
    term when ``locus_distance`` is given, and for the logit propensity score). Columns: ``estimator, OR, lo, hi,
    P, interval, converged``. ``attrs`` carries counts, the propensity c-statistic and the largest |SMD| before
    and after quintile weighting.
    """
    D = df.reset_index(drop=True)
    check_table(D, outcome, exposure, covariates)
    y = D[outcome].values.astype(float)
    e = D[exposure].values.astype(float)
    ps = fit_ps(D, exposure, covariates, linear)
    bal, psq = balance(D, exposure, covariates, ps.ps)
    rows = []

    def rec(name, o, lo, hi, p, interval, conv=True):
        rows.append(dict(estimator=name, OR=o, lo=lo, hi=hi, P=p, interval=interval, converged=bool(conv)))

    fr = naive(y, e)
    rec("naive (Fisher)", fr.odds_ratio, fr.lo, fr.hi, fr.pvalue, "exact 95%")
    r = g1(y, e, ps.logit, profile_levels=profile_levels, wald_levels=wald_levels, maxit=maxit)
    for lev in profile_levels:
        lo, hi = r.profile_or(lev)
        rec("G1", r.odds_ratio, lo, hi, r.pvalue, f"profile {lev:.1%}", r.converged)
    for lev in wald_levels:
        lo, hi = r.wald_or(lev)
        rec("G1", r.odds_ratio, lo, hi, r.pvalue, f"Wald {lev:.1%}", r.converged)
    o, lo, hi, p = g1b(y, e, np.asarray(psq))
    rec("G1b (PS-quintile Mantel-Haenszel)", o, lo, hi, p, "RBG 95%")
    X_oe = np.c_[np.ones(len(D)), D[list(covariates)].values.astype(float)]
    gc = g1c(y, e, X_oe, maxit=maxit)
    oe, p2, n_obs, n_exp = gc
    rec(f"G1c (observed/expected; O={n_obs}, E={n_exp:.1f})", oe, np.nan, np.nan, p2, "", gc.converged)
    inl, iid, ld = locus_arrays(D, locus, locus_distance)
    if inl is not None:
        cells = [((e == a) & (inl == b)).sum() for a in (0, 1) for b in (False, True)]
        if min(cells) > 0:
            hr = h_interaction(y, e, inl, ps.logit, maxit=maxit)
            rec("H (exposure x in-locus)", *hr, "profile 95%", hr.converged)
        else:              # the interaction needs exposed and unexposed genes both inside and outside loci
            rec("H (exposure x in-locus): not estimable", np.nan, np.nan, np.nan, np.nan, "", False)
        cov = ps.logit[inl][:, None] if ld is None else np.c_[ld[inl], ps.logit[inl]]
        wr = w1(y[inl], e[inl], iid[inl], cov) if _w1_identified(y[inl], e[inl], iid[inl]) else None
        if wr is not None:
            rec(f"W1 (within-locus; {wr.informative_strata} informative strata)", wr.odds_ratio, float(np.exp(wr.lo)),
                float(np.exp(wr.hi)), wr.pvalue, "profile 95%", wr.converged)
        else:
            rec("W1 (within-locus): not estimable, no locus has both events and non-events and both exposed and "
                "unexposed genes", np.nan, np.nan, np.nan, np.nan, "", False)
    out = pd.DataFrame(rows)
    out.attrs.update(n=len(D), exposed=int(e.sum()), events_exposed=int(y[e == 1].sum()),
                     events_other=int(y[e == 0].sum()), ps_c=float(ps.c_statistic), ps_converged=ps.converged,
                     max_abs_smd_raw=float(bal.smd_raw.abs().max()), max_abs_smd_after=float(bal.smd_psq.abs().max()))
    return out


@dataclass
class Calibration:
    scenarios: list
    replicates: pd.DataFrame
    summary: pd.DataFrame
    calibrated: bool | None        # None: N0 and N1 were not both simulated
    generator: np.ndarray
    generator_converged: bool


def calibrate(df: pd.DataFrame, outcome: str, exposure: str, covariates, generator_covariates, linear=(),
              locus=None, locus_distance=None, targeting="estimand", n_null=2000, n_power=1000,
              nulls=("N0", "N1", "N2"), power_ors=(1.5, 2.0, 2.7), seed=None, estimators=("g1", "naive"),
              workers=1, maxit=DEFAULT_MAXIT, threshold=VALIDITY_THRESHOLD, max_failed=0.05) -> Calibration:
    """Plasmode calibration of G1 for this exposure, before its outcome is read.

    The generator ``Y ~ [1, generator_covariates, L, L x locus_distance]`` is fitted on the unexposed genes only
    (the locus terms only when ``locus`` is given; the distance term only when ``locus_distance`` is given; N2
    needs ``locus``). Under N1 the ``generator_covariates`` coefficients are doubled. Scenarios are
    estimand-targeted by default (A6). ``estimators`` may include "g1", "naive", "g1b", "g1c", "equiv",
    "g1_profile" and, with ``locus``, "h" and "w1".

    ``calibrated`` applies the registered validity rule (type I error <= ``threshold`` under N0 and N1) and, in
    addition, requires at most ``max_failed`` failed G1 fits in each; it is None unless N0 and N1 were both run.
    ``seed`` is required.
    """
    if seed is None:
        raise ValueError("a seed is required, so that the calibration can be reproduced")
    D = df.reset_index(drop=True)
    check_table(D, outcome, exposure, list(covariates) + list(generator_covariates), outcome_complete=False)
    e = D[exposure].values.astype(float)
    y = np.where(e == 0, D[outcome].values.astype(float), np.nan)          # exposed outcomes are never read
    ps = fit_ps(D, exposure, covariates, linear)
    X_g1 = g1_design(ps.logit, e)
    inl, iid, ld = locus_arrays(D, locus, locus_distance)
    cols = [np.ones(len(D))] + [D[c].values.astype(float) for c in generator_covariates]
    if inl is None:
        bad = {"N2", "h", "w1"} & (set(nulls) | set(estimators))
        if bad:
            raise ValueError(f"{sorted(bad)} need locus columns (locus=...)")
        inl = np.zeros(len(D), bool)
        iid = np.full(len(D), -1)
    else:
        cols.append(inl.astype(float))
        if ld is not None:
            cols.append(ld)
    X_gen = np.column_stack(cols)
    gf = firth_fit(X_gen[e == 0], y[e == 0], maxit=maxit)
    gam = gf.beta
    k = len(generator_covariates)
    eta0, eta1, shift = _pm.null_scenarios(X_gen, gam, inl, doubled=tuple(range(1, k + 1)))
    scen = _pm.build_scenarios(X_g1, eta0, eta1, shift, e, inl, n_null=n_null, n_power=n_power, nulls=nulls,
                               power_ors=power_ors, targeting=targeting)
    _, psq = balance(D, exposure, covariates, ps.ps)
    extra = dict(ps_quintile=np.asarray(psq), X_oe=np.c_[np.ones(len(D)), D[list(covariates)].values.astype(float)])
    if locus is not None:
        wcov = ps.logit[:, None] if ld is None else np.c_[ld, ps.logit]
        extra.update(X_h=h_design(ps.logit, e, inl), X_w1=np.c_[e, wcov][inl])
    design = _pm.PlasmodeDesign(E=e, X_g1=X_g1, in_locus=inl, locus_id=iid, maxit=maxit, **extra)
    R = _pm.run(design, scen, seed, estimators=estimators, workers=workers)
    S = _pm.summarise(R, targets={s.name: s.target_log_or for s in scen})
    ok = (_pm.validity(S, "g1", ("N0", "N1"), threshold, max_failed)
          if {"N0", "N1"} <= set(nulls) and "g1" in estimators else None)
    if not gf.converged:
        warnings.warn("the plasmode generator did not converge; see Calibration.generator_converged", UserWarning,
                      stacklevel=2)
    return Calibration(scen, R, S, ok, gam, gf.converged)


def matched_controls(df: pd.DataFrame, outcome: str, exposure: str, covariates, linear=(), n_sets=1000, seed=None,
                     workers=1, maxit=DEFAULT_MAXIT, band=NC_BAND):
    """Discoverability-matched negative control (see :mod:`nomibench.negative_control`). Returns
    ``(per-set table, summary)``; ``summary.passes`` applies the band. ``seed`` is required. Only the unexposed
    genes' outcomes are read, so the exposed genes' outcomes may be missing."""
    if seed is None:
        raise ValueError("a seed is required, so that the control can be reproduced")
    D = df.reset_index(drop=True)
    check_table(D, outcome, exposure, covariates, outcome_complete=False)
    R = _nc.run(D, outcome, exposure, covariates, linear, n_sets=n_sets, seed=seed, workers=workers, maxit=maxit)
    return R, _nc.summarise(R, band=band)
