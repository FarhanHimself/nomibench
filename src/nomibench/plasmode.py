"""Plasmode simulation for calibrating enrichment estimators (Franklin et al. 2014, Comput Stat Data Anal 72:219).

The real covariates, exposure and locus structure are kept; only the outcome is simulated, from a generator fitted
to the real outcome of the UNEXPOSED genes (the exposed genes' outcomes never enter).

Generator (paper 1, spec v1 section D): Firth logistic ``Y ~ [1, discoverability terms, L, L * log10 D]`` fitted on
the unexposed genes, where L flags genes inside a GWAS locus interval and D is the distance to the interval's
nearest index variant.

Null scenarios:
  N0  the fitted generator;
  N1  the discoverability coefficients doubled, intercept re-solved to keep the mean event rate;
  N2  N0 plus a normal random intercept per locus interval (SD ``csd``), shifted so the in-locus marginal event
      rate is unchanged (40-point Gauss-Hermite).
Power scenarios add ``log(OR) * E`` to N0's linear predictor.

Estimand targeting (paper 1 amendment A6). G1's population estimand is the exposure coefficient of the logistic
projection of the scenario's marginal P(Y = 1) onto G1's design ``[1, E, s(logit PS)]``. Generator terms that are
not in the propensity model (here L and L * log10 D) make ``beta = 0`` a non-null for G1's estimand whenever the
exposure is enriched in loci. With ``targeting="estimand"`` each scenario's exposure coefficient beta* is solved
so that the estimand equals the scenario's target (0 for the nulls; log OR for the power scenarios).
``targeting="none"`` reproduces the untargeted scenarios (paper 1 step 19, and A5 as written).

Arithmetic as registered (``gwcore.py``, ``20b_e5_calibration.py``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from ._util import expit
from .conditional import cfl_fit, make_strata
from .firth import DEFAULT_MAXIT, firth_fit, firth_lrt, firth_profile_ci, wald_se
from .tables import fisher_logor, mh_rbg, poibin_two_sided, strat_counts

__all__ = ["fit_generator", "null_scenarios", "marginal_p", "g1_target", "beta_star", "Scenario", "build_scenarios",
           "simulate", "PlasmodeDesign", "replicate", "run", "summarise", "validity", "ESTIMATORS",
           "seeds_from_root", "LN2"]

LN2 = np.log(2.0)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / GH_W.sum()


# ----------------------------------------------------------------------------------------------- generator
def fit_generator(X_gen, y, exposed, maxit=DEFAULT_MAXIT):
    """Firth logistic generator fitted on the unexposed genes only. ``X_gen`` includes the intercept.
    ``y`` may hold NaN for exposed genes (they are never read)."""
    X_gen = np.asarray(X_gen, float)
    m = np.asarray(exposed) == 0
    return firth_fit(X_gen[m], np.asarray(y)[m], maxit=maxit).beta


def null_scenarios(X_gen, gam, in_locus, doubled=(1, 2, 3), csd=1.0):
    """Linear predictors of N0 and N1 and the N2 shift.

    ``doubled`` indexes the generator coefficients multiplied by 2 under N1 (paper 1: the three discoverability
    terms). Returns ``(eta0, eta1, shift)``.
    """
    X_gen = np.asarray(X_gen, float)
    inl = np.asarray(in_locus, bool)
    eta0 = X_gen @ gam
    p0 = expit(eta0)
    g1 = gam.copy()
    idx = np.asarray(doubled, int)
    g1[idx] *= 2
    g1[0] = brentq(lambda c: expit(X_gen @ np.r_[c, g1[1:]]).mean() - p0.mean(), -30, 30)
    eta1 = X_gen @ g1
    if not inl.any():                    # no locus structure: N2 is undefined
        return eta0, eta1, 0.0
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(40)
    gh_w = gh_w / gh_w.sum()
    tgt = p0[inl].mean()
    shift = brentq(lambda c: np.mean([np.sum(gh_w * expit(e_ + csd * gh_x + c)) for e_ in eta0[inl]]) - tgt, -5, 5)
    return eta0, eta1, shift


# ----------------------------------------------------------------------------------------------- A6 targeting
def marginal_p(eta, in_locus, csd, shift):
    """Marginal P(Y = 1) per gene; under N2 the locus random intercept is integrated out (Gauss-Hermite)."""
    p = expit(eta)
    if csd > 0:
        inl = np.asarray(in_locus, bool)
        p = p.copy()
        p[inl] = (expit(eta[inl][:, None] + csd * GH_X[None, :] + shift) * GH_W[None, :]).sum(axis=1)
    return p


def g1_target(X, p, tol=1e-12):
    """Exposure coefficient of the logistic projection of ``p`` onto ``X`` = [1, E, s(logit PS)]: the solution of
    the score equations X'(p - mu) = 0. This is G1's population estimand under a scenario with marginal ``p``."""
    b = np.zeros(X.shape[1])
    for _ in range(200):
        mu = expit(X @ b)
        W = mu * (1 - mu)
        st = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (p - mu))
        b += st
        if np.max(np.abs(st)) < tol:
            return float(b[1])
    raise RuntimeError("g1_target did not converge")


def beta_star(X, eta, E, in_locus, csd, shift, target):
    """Exposure coefficient beta* such that G1's estimand under ``eta + beta* E`` equals ``target``."""
    def f(bb):
        return g1_target(X, marginal_p(eta + bb * E, in_locus, csd, shift)) - target

    return float(brentq(f, -10, 10, xtol=1e-12))


@dataclass
class Scenario:
    name: str
    eta: np.ndarray            # linear predictor, exposure coefficient included
    n_reps: int
    csd: float = 0.0           # SD of the locus random intercept (N2)
    shift: float = 0.0         # N2 intercept shift for in-locus genes
    target_log_or: float = 0.0
    beta: float = 0.0          # exposure coefficient used
    target_or_as_written: float = float("nan")   # G1's estimand if beta = target (untargeted)


def build_scenarios(X_g1, eta0, eta1, shift, E, in_locus, n_null=2000, n_power=1000, nulls=("N0", "N1", "N2"),
                    power_ors=(1.5, 2.0, 2.7), targeting="estimand", csd=1.0, power_names=None):
    """Scenario list in a fixed order: the nulls, then the power scenarios.

    ``targeting="estimand"`` (A6) solves beta* per scenario; ``"none"`` uses beta = target (untargeted).
    ``power_names`` maps OR -> scenario name (default ``"P-GW_OR{or}"``).
    """
    E = np.asarray(E, float)
    base = {"N0": (eta0, 0.0, 0.0, 0.0), "N1": (eta1, 0.0, 0.0, 0.0), "N2": (eta0, csd, shift, 0.0)}
    names = power_names or {o: f"P-GW_OR{o}" for o in power_ors}
    spec = [(n, *base[n], n_null) for n in nulls] + [(names[o], eta0, 0.0, 0.0, float(np.log(o)), n_power)
                                                     for o in power_ors]
    out = []
    for name, eb, cs, sh, tgt, nrep in spec:
        th_asw = g1_target(X_g1, marginal_p(eb + tgt * E, in_locus, cs, sh))
        if targeting == "estimand":
            bst = beta_star(X_g1, eb, E, in_locus, cs, sh, tgt)
            th = g1_target(X_g1, marginal_p(eb + bst * E, in_locus, cs, sh))
            if abs(th - tgt) > 1e-8:
                raise RuntimeError(f"{name}: estimand targeting failed ({th} vs {tgt})")
        elif targeting == "none":
            bst = tgt
        else:
            raise ValueError("targeting must be 'estimand' or 'none'")
        out.append(Scenario(name, eb + bst * E, nrep, cs, sh, tgt, bst, float(np.exp(th_asw))))
    return out


# ----------------------------------------------------------------------------------------------- simulation
def simulate(eta, in_locus, locus_id, csd, shift, seed):
    """One simulated outcome vector. Draw order is fixed (N2 locus intercepts, then Y), so every estimator sees
    the identical replicate. Returns ``(y, rng)``; the generator continues for estimators that need draws."""
    rng = np.random.default_rng(seed)
    if csd > 0:
        iid = np.asarray(locus_id, int)
        u = rng.normal(0, csd, iid.max() + 2)
        eta = eta + np.where(in_locus, u[iid] + shift, 0.0)
    y = rng.binomial(1, expit(eta)).astype(float)
    return y, rng


@dataclass
class PlasmodeDesign:
    """Everything a replicate's estimators need, fixed across replicates.

    ``E`` exposure; ``X_g1`` G1 design; ``in_locus``, ``locus_id`` (integer, -1 outside loci) for N2 and W1;
    optional: ``ps_quintile`` (G1b), ``X_oe`` (G1c), ``X_h`` (H), ``X_w1`` (W1 covariates, in-locus rows only).
    """
    E: np.ndarray
    X_g1: np.ndarray
    in_locus: np.ndarray
    locus_id: np.ndarray
    ps_quintile: np.ndarray | None = None
    X_oe: np.ndarray | None = None
    X_h: np.ndarray | None = None
    X_w1: np.ndarray | None = None
    maxit: int = DEFAULT_MAXIT
    margin: float = 2.0
    z_equiv: float = float(norm.ppf(0.975))


def _est_g1(y, d):
    t = firth_lrt(d.X_g1, y, 1, maxit=d.maxit)
    se = wald_se(d.X_g1, t.beta, 1)
    return dict(g1_logor=float(t.beta[1]), g1_p=float(t.pvalue), g1_se=se)


def _est_g1_profile(y, d):
    f = firth_fit(d.X_g1, y, maxit=d.maxit)
    ci = firth_profile_ci(d.X_g1, y, 1, f.beta, f.loglik, level=0.95, maxit=d.maxit)
    return dict(g1_prof_lo=float(ci.lo), g1_prof_hi=float(ci.hi))


def _est_equiv(y, d):
    """Is G1's Wald upper bound below the margin? Stored as the event itself (1 = upper bound < margin)."""
    f = firth_fit(d.X_g1, y, maxit=d.maxit)
    se = wald_se(d.X_g1, f.beta, 1)
    return dict(equiv_logor=float(f.beta[1]), upper_lt_margin=float(f.beta[1] + d.z_equiv * se < np.log(d.margin)))


def _est_naive(y, d):
    """Fisher's exact P with the log sample (cross-product) odds ratio, as registered; the real-data
    :func:`nomibench.estimators.naive` reports the conditional MLE instead (the P value is the same)."""
    lo, p = fisher_logor(y, d.E)
    return dict(fisher_logor=lo, fisher_p=p)


def _est_g1b(y, d):
    o, lo, hi, p = mh_rbg(*strat_counts(y, d.E, d.ps_quintile))
    return dict(g1b_logor=(np.log(o) if o and o > 0 else np.nan), g1b_p=p)


def _est_g1c(y, d):
    a = d.E
    Bm = a == 0
    b = firth_fit(d.X_oe[Bm], y[Bm], maxit=d.maxit).beta
    pA = expit(d.X_oe[a == 1] @ b)
    n_obs = int(y[a == 1].sum())
    n_exp = pA.sum()
    p2, _ = poibin_two_sided(pA, n_obs)
    return dict(g1c_logoe=(np.log(n_obs / n_exp) if n_obs > 0 else np.nan), g1c_p=p2)


def _est_h(y, d):
    t = firth_lrt(d.X_h, y, 3, maxit=d.maxit)
    return dict(h_logor=float(t.beta[3]), h_p=float(t.pvalue))


def _est_w1(y, d):
    inl = np.asarray(d.in_locus, bool)
    try:
        st = make_strata(d.X_w1, y[inl], np.asarray(d.locus_id)[inl])
        if len(st) == 0:
            return dict(w1_logor=np.nan, w1_p=np.nan)      # no informative stratum: counted as a non-rejection
        p_ = d.X_w1.shape[1]
        b, lp = cfl_fit(st, p_, warn=False)
        b0 = b.copy()
        b0[0] = 0
        _, lp0 = cfl_fit(st, p_, fixed={0: 0.0}, beta0=b0, warn=False)
        return dict(w1_logor=float(b[0]), w1_p=float(chi2.sf(max(2 * (lp - lp0), 0), 1)))
    except Exception:
        return dict(w1_logor=np.nan, w1_p=np.nan)


ESTIMATORS: dict[str, Callable] = {"g1": _est_g1, "g1_profile": _est_g1_profile, "equiv": _est_equiv,
                                   "naive": _est_naive, "g1b": _est_g1b, "g1c": _est_g1c, "h": _est_h, "w1": _est_w1}


def replicate(scenario: Scenario, rep: int, seed, design: PlasmodeDesign, estimators=("g1", "naive")):
    """Simulate one replicate and apply the estimators. A failed estimator returns NaN (a non-rejection)."""
    y, _ = simulate(scenario.eta, design.in_locus, design.locus_id, scenario.csd, scenario.shift, seed)
    row = dict(scenario=scenario.name, rep=rep, ysum=float(y.sum()), yexp=float(y[np.asarray(design.E) == 1].sum()))
    for name in estimators:
        try:
            row.update(ESTIMATORS[name](y, design))
        except Exception:
            row.update({k: np.nan for k in _NAN_KEYS[name]})
    return row


_NAN_KEYS = {"g1": ("g1_logor", "g1_p", "g1_se"), "g1_profile": ("g1_prof_lo", "g1_prof_hi"),
             "equiv": ("equiv_logor", "upper_lt_margin"), "naive": ("fisher_logor", "fisher_p"),
             "g1b": ("g1b_logor", "g1b_p"), "g1c": ("g1c_logoe", "g1c_p"), "h": ("h_logor", "h_p"),
             "w1": ("w1_logor", "w1_p")}

_G = {}


def _init(bundle):
    _G.clear()
    _G.update(bundle)


def _job(args):
    sc, k, seed = args
    return replicate(_G["scen"][sc], k, seed, _G["design"], _G["estimators"])


def _fresh(root):
    """A SeedSequence with no children spawned yet: the root's own copy, so reusing a root gives the same streams."""
    if isinstance(root, np.random.SeedSequence):
        return np.random.SeedSequence(root.entropy, spawn_key=root.spawn_key, pool_size=root.pool_size)
    if root is None:
        raise ValueError("a seed is required, so that the run can be reproduced")
    return np.random.SeedSequence(root)


def seeds_from_root(root, scenarios: Sequence[Scenario]):
    """Default seeding: ``SeedSequence(root).spawn(len(scenarios))``, then each scenario child spawns one
    child per replicate (paper 1 amendments A5/A6 and A10). A scenario's stream depends on its position in the
    scenario list, not on any scenario's replicate count."""
    ss = _fresh(root)
    return {s.name: list(ch.spawn(s.n_reps)) for s, ch in zip(scenarios, ss.spawn(len(scenarios)))}


def run(design: PlasmodeDesign, scenarios: Sequence[Scenario], seeds, estimators=("g1", "naive"), workers=1,
        chunksize=8) -> pd.DataFrame:
    """Run every replicate of every scenario; one row per replicate.

    ``seeds`` is a root seed (int or SeedSequence; see :func:`seeds_from_root`) or a mapping
    ``{scenario name: [one seed per replicate]}`` for full control (e.g. to reproduce a registered seed tree).
    Results do not depend on ``workers``.
    """
    if not isinstance(seeds, dict):
        seeds = seeds_from_root(seeds, scenarios)
    jobs = [(s.name, k, sd) for s in scenarios for k, sd in enumerate(seeds[s.name][:s.n_reps])]
    bundle = dict(scen={s.name: s for s in scenarios}, design=design, estimators=tuple(estimators))
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(bundle,)) as pool:
            rows = pool.map(_job, jobs, chunksize=chunksize)
    else:
        _init(bundle)
        rows = [_job(j) for j in jobs]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- summaries
def _verdict(rate, band=(0.035, 0.065), failed=0.0):
    if failed >= 0.5:                            # mostly failed fits: the rejection rate means nothing
        return "FAILED FITS"
    return "ANTI-CONSERVATIVE" if rate > band[1] else "conservative" if rate < band[0] else "calibrated"


def classify_log(lo, hi, margin=2.0):
    """Four-way reading on the log scale, vectorised (NaN -> 'failed')."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    lm = np.log(margin)
    out = np.full(len(lo), "failed", dtype=object)
    ok = np.isfinite(lo) & np.isfinite(hi)
    out[ok & (lo > 0) & (hi >= lm)] = "enriched"
    out[ok & (lo > 0) & (hi < lm)] = "modest"
    out[ok & (lo <= 0) & (hi < lm)] = "no_material"
    out[ok & (lo <= 0) & (hi >= lm)] = "inconclusive"
    return out


def summarise(R: pd.DataFrame, targets: dict | None = None, alpha=0.05, band=(0.035, 0.065), margin=2.0,
              z95=None, z975=None) -> pd.DataFrame:
    """Per scenario and estimator: rejection rate at P < alpha (a failed fit counts as a non-rejection), Monte
    Carlo SE, failure rate, mean estimate and, for G1, the decision operating characteristics on its Wald
    intervals (P(lower > 1), P(upper < margin), four-way reading shares) and the bias against ``targets``.
    Null scenarios (names starting with "N") get a calibration verdict against ``band``."""
    with warnings.catch_warnings():                 # all-NaN estimates (e.g. no informative stratum) give NaN means
        warnings.simplefilter("ignore", RuntimeWarning)
        return _summarise(R, targets, alpha, band, margin, z95, z975)


def _summarise(R, targets, alpha, band, margin, z95, z975):
    z95 = norm.ppf(0.975) if z95 is None else z95
    z975 = norm.ppf(0.9875) if z975 is None else z975
    lm = np.log(margin)
    rows = []
    pairs = [("g1", "g1_logor", "g1_p"), ("naive", "fisher_logor", "fisher_p"), ("g1b", "g1b_logor", "g1b_p"),
             ("g1c", "g1c_logoe", "g1c_p"), ("h", "h_logor", "h_p"), ("w1", "w1_logor", "w1_p")]
    for sc, g in R.groupby("scenario", sort=False):
        null = str(sc).startswith("N")
        for est, ce, cp in pairs:
            if cp not in g:
                continue
            n = len(g)
            pv = g[cp].values.astype(float)
            rej = float(np.mean(np.nan_to_num(pv, nan=1.0) < alpha))
            est_vals = g[ce].values.astype(float)
            failed = float(np.mean(np.isnan(pv)))
            r = dict(scenario=sc, estimator=est, reps=n, reject=rej, mc_se=float(np.sqrt(rej * (1 - rej) / n)),
                     failed=failed, mean_estimate=float(np.nanmean(np.where(np.isfinite(est_vals), est_vals, np.nan))),
                     verdict=_verdict(rej, band, failed) if null else "")
            if est == "g1" and "g1_se" in g:
                b, se = est_vals, g.g1_se.values.astype(float)
                with np.errstate(invalid="ignore"):
                    lo95, hi95, lo975, hi975 = b - z95 * se, b + z95 * se, b - z975 * se, b + z975 * se
                cls = pd.Series(classify_log(lo975, hi975, margin)).value_counts(normalize=True)
                r.update(lower95_gt1=float(np.mean(lo95 > 0)), lower975_gt1=float(np.mean(lo975 > 0)),
                         upper95_lt_margin=float(np.mean(hi95 < lm)), upper975_lt_margin=float(np.mean(hi975 < lm)),
                         **{f"cls_{k}": float(cls.get(k, 0)) for k in
                            ("enriched", "modest", "no_material", "inconclusive", "failed")})
                if targets is not None and sc in targets:
                    r.update(target_logor=targets[sc], mean_minus_target=float(np.nanmean(b)) - targets[sc])
            rows.append(r)
        # Decision operating characteristics. The "reject" column holds the share of replicates whose bound falls
        # below the margin; a failed fit or an unbracketed bound counts as not below it, except in the
        # *_failed_as_excl row, which counts it as below (A10 rule 2's cautious reading of false exclusion).
        if "upper_lt_margin" in g:
            v = np.nan_to_num(g.upper_lt_margin.values.astype(float), nan=0.0)
            rows.append(dict(scenario=sc, estimator="g1_wald_upper_lt_margin", reps=len(g), reject=float(v.mean()),
                             mc_se=float(np.sqrt(v.mean() * (1 - v.mean()) / len(g))),
                             failed=float(np.mean(np.isnan(g.upper_lt_margin.values.astype(float)))),
                             mean_estimate=float(np.nanmean(g.equiv_logor)), verdict=""))
        if "g1_prof_hi" in g:
            hi = g.g1_prof_hi.values.astype(float)
            lo = g.g1_prof_lo.values.astype(float)
            fail = np.isnan(hi)
            row = dict(scenario=sc, estimator="g1_profile_upper_lt_margin", reps=len(g), reject=float(np.mean(hi < lm)),
                       mc_se=np.nan, failed=float(np.mean(fail)), mean_estimate=np.nan, verdict="")
            if targets is not None and sc in targets:
                t = targets[sc]
                row["coverage"] = float(np.mean((lo <= t) & (t <= hi)))
            rows.append(row)
            rows.append(dict(scenario=sc, estimator="g1_profile_upper_lt_margin_failed_as_excl", reps=len(g),
                             reject=float(np.mean((hi < lm) | fail)), mc_se=np.nan, failed=float(np.mean(fail)),
                             mean_estimate=np.nan, verdict=""))
    return pd.DataFrame(rows)


def validity(summary: pd.DataFrame, estimator="g1", scenarios=("N0", "N1"), threshold=0.065,
             max_failed=None) -> bool:
    """Registered validity rule: the estimator is calibrated if its type I error is <= ``threshold`` under each
    of ``scenarios``. Raises if a required scenario was not simulated. With ``max_failed``, it is also not
    calibrated if more than that share of fits failed in any of them (not part of the registered rule)."""
    s = summary[summary.estimator == estimator].set_index("scenario")
    missing = [n for n in scenarios if n not in s.index]
    if missing or not scenarios:
        raise ValueError(f"validity needs the null scenarios {list(scenarios)}; not simulated: {missing}")
    ok = all(s.loc[n, "reject"] <= threshold for n in scenarios)
    if max_failed is not None:
        ok = ok and all(s.loc[n, "failed"] <= max_failed for n in scenarios)
    return bool(ok)
