"""D1: doubly robust enrichment estimate, cross-fitted AIPW for the ATT risk ratio.

Estimand: RR_ATT = P(Y = 1 | E = 1) / E[ mu0(X) | E = 1 ], mu0(X) = P(Y = 1 | X, E = 0).

Estimator (paper 1 amendment A4 section 10.1):
  * augmented inverse-probability weighting for the ATT with odds weights e/(1 - e) (e capped at 0.99);
  * nuisances e(X) and mu0(X) cross-fitted over 5 folds stratified on E (Chernozhukov et al. 2018,
    Econometrics J 21:C1, doi:10.1111/ectj.12097);
  * each nuisance is a two-member super learner: the registered spline logistic model (unpenalised ML) and
    gradient-boosted trees, combined with a convex weight chosen by log-loss on an 80/20 split of the
    training fold (van der Laan, Polley & Hubbard 2007);
  * influence-function SE on log RR, Wald CI and P;
  * the estimate is repeated over independent fold splits and combined by the median rule (Chernozhukov 2018):
    theta = median(theta_s), sigma^2 = median(sigma_s^2 + (theta_s - theta)^2).

Arithmetic and seeds as registered (``19c_d1.py``, ``20c_e5_outcome.py``). Gradient boosting runs single-threaded
(as the registered runs did), so results do not depend on the machine's core count.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from ._util import check_table
from .propensity import spline_formula

__all__ = ["logit_mle", "super_learner", "folds", "crossfit_e", "crossfit_mu0", "aipw_att", "wald_p", "basis",
           "split_seeds", "D1Result", "d1", "registered_mu0_seed", "outcome_checksum", "d1_replicate"]

GRID = np.linspace(0, 1, 21)


def logit_mle(X, y, maxit=50):
    """Unpenalised logistic regression by Newton-Raphson with step halving (1e-8 ridge for numerical stability)."""
    with np.errstate(over="ignore"):            # exp overflow gives p = 0 exactly, as in the registered runs
        return _logit_mle(X, y, maxit)


def _logit_mle(X, y, maxit):
    b = np.zeros(X.shape[1])
    ll_old = -np.inf
    for _ in range(maxit):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        t = 1.0
        while True:
            nb = b + t * step
            e2 = X @ nb
            ll = np.sum(y * e2 - np.logaddexp(0, e2))
            if ll >= ll_old - 1e-10 or t < 1e-4:
                break
            t /= 2
        conv = abs(ll - ll_old) < 1e-10
        b, ll_old = nb, ll
        if conv:
            break
    return b


def _hgb(seed):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, early_stopping=True,
                                          validation_fraction=0.2, n_iter_no_change=20, min_samples_leaf=50,
                                          l2_regularization=1.0, random_state=int(seed % (2**31 - 1)))


def _logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))


def super_learner(Btr, Xtr, ytr, Bte, Xte, seed):
    """Convex weight on the spline GLM, chosen on an 80/20 split of the training data; both members then refitted
    on all training data. Returns ``(predictions, GLM weight)``."""
    with np.errstate(over="ignore"):
        return _super_learner(Btr, Xtr, ytr, Bte, Xte, seed)


def _super_learner(Btr, Xtr, ytr, Bte, Xte, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    cut = int(0.8 * len(idx))
    a, b = idx[:cut], idx[cut:]
    if ytr[a].sum() < 2 or ytr[a].sum() > len(a) - 2:          # degenerate split: GLM only
        w = 1.0
    else:
        g = logit_mle(Btr[a], ytr[a])
        pg = 1 / (1 + np.exp(-(Btr[b] @ g)))
        ph = _hgb(seed).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        w = GRID[int(np.argmin([_logloss(ytr[b], v * pg + (1 - v) * ph) for v in GRID]))]
    g = logit_mle(Btr, ytr)
    pg = 1 / (1 + np.exp(-(Bte @ g)))
    if w < 1.0:
        ph = _hgb(seed + 1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        return w * pg + (1 - w) * ph, w
    return pg, w


def folds(E, split_seed):
    """Five folds stratified on the exposure."""
    return list(StratifiedKFold(5, shuffle=True, random_state=int(split_seed % (2**31 - 1))).split(np.zeros(len(E)), E))


def crossfit_e(B, X, E, fl, seed):
    e_hat = np.zeros(len(E))
    ws = []
    for k, (tr, te) in enumerate(fl):
        e_hat[te], w = super_learner(B[tr], X[tr], E[tr], B[te], X[te], seed + 17 * k)
        ws.append(w)
    return e_hat, ws


def crossfit_mu0(B, X, E, Y, fl, seed):
    mu = np.zeros(len(E))
    ws = []
    for k, (tr, te) in enumerate(fl):
        t0 = tr[E[tr] == 0]
        mu[te], w = super_learner(B[t0], X[t0], Y[t0], B[te], X[te], seed + 31 * k)
        ws.append(w)
    return mu, ws


def aipw_att(Y, E, e_hat, mu0):
    """AIPW estimate of log RR_ATT and its influence-function SE."""
    e_hat = np.minimum(e_hat, 0.99)
    n1 = E.sum()
    pi = n1 / len(E)
    psi1 = (E * Y).sum() / n1
    w = e_hat / (1 - e_hat)
    psi0 = (E * mu0 + (1 - E) * w * (Y - mu0)).sum() / n1
    if psi1 <= 0 or psi0 <= 0:
        return np.nan, np.nan
    IF = (E * (Y - psi1) / pi) / psi1 - ((E * (mu0 - psi0) + (1 - E) * w * (Y - mu0)) / pi) / psi0
    return np.log(psi1 / psi0), np.sqrt((IF ** 2).sum()) / len(Y)


def wald_p(theta, se):
    if not np.isfinite(theta) or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2 * norm.sf(abs(theta) / se)


def basis(df: pd.DataFrame, cols, linear=()):
    """The registered spline basis of the covariates (intercept included), used by the GLM learner."""
    return dmatrix(spline_formula(df, cols, linear), df, return_type="dataframe").values


def split_seeds(entropy, splits=50):
    """Registered split seeds: ``SeedSequence(entropy).spawn(splits)``, one 32-bit state word each."""
    return [int(c.generate_state(1)[0]) for c in np.random.SeedSequence(entropy).spawn(splits)]


@dataclass
class D1Result:
    rr: float
    lo: float
    hi: float
    pvalue: float
    log_rr: float
    se: float
    splits_used: int
    split_sd: float           # SD of log RR across splits
    exposed: int
    events_exposed: int


_G = {}


def _init(bundle):
    _G.clear()
    _G.update(bundle)


def _one_split(args):
    k, s = args
    with threadpool_limits(limits=1):
        B, X, E, y = _G["B"], _G["X"], _G["E"], _G["y"]
        fl = folds(E, s)
        eh, _ = crossfit_e(B, X, E, fl, s)
        mu0, _ = crossfit_mu0(B, X, E, y, fl, _G["mu0_seed"](k))
        return aipw_att(y, E, eh, mu0)


def registered_mu0_seed(k):
    """Seed of the outcome-model super learner in split k (paper 1: 1,000,003 x (k + 1))."""
    return 1_000_003 * (k + 1)


def d1(df: pd.DataFrame, outcome: str, exposure: str, cols, linear=(), splits=50, entropy=None, seeds=None,
       mu0_seed=registered_mu0_seed, z=None, workers=1) -> D1Result:
    """D1 on the real outcome: cross-fitted AIPW ATT risk ratio, median rule over ``splits`` fold splits.

    Split seeds come from ``seeds`` or ``split_seeds(entropy, splits)``. Paper 1 used entropy
    ``[20260927, 1]`` (E1), ``[20260927, 1, 2]`` (E1w), ``[20260927, 1, 3]`` (E2) and ``[20260928, 1, k]``
    for the k-th E5 set, with ``z = 1.959964`` in step 19 and the exact normal quantile in step 20.
    """
    check_table(df, outcome, exposure, cols)
    if seeds is None:
        if entropy is None:
            raise ValueError("give `entropy` or explicit `seeds`, so that the fold splits can be reproduced")
        seeds = split_seeds(entropy, splits)
    E = df[exposure].values.astype(float)
    y = df[outcome].values.astype(float)
    bundle = dict(B=basis(df, cols, linear), X=df[list(cols)].values.astype(float), E=E, y=y, mu0_seed=mu0_seed)
    jobs = list(enumerate(seeds))
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(bundle,)) as pool:
            r = pool.map(_one_split, jobs, chunksize=1)
    else:
        _init(bundle)
        r = [_one_split(j) for j in jobs]
    th = np.array([t for t, _ in r])
    v = np.array([s_ ** 2 for _, s_ in r])
    tm = np.nanmedian(th)
    sm_ = np.sqrt(np.nanmedian(v + (th - tm) ** 2))
    z = norm.ppf(0.975) if z is None else z
    return D1Result(float(np.exp(tm)), float(np.exp(tm - z * sm_)), float(np.exp(tm + z * sm_)), wald_p(tm, sm_),
                    float(tm), float(sm_), int(np.isfinite(th).sum()), float(np.nanstd(th)), int(E.sum()),
                    int(y[E == 1].sum()))


def outcome_checksum(y):
    """Order-sensitive checksum of a 0/1 outcome vector (registered: sum of 1-based positions of events, mod
    1,000,003); used to confirm two runs simulated the same replicate."""
    y = np.asarray(y, float)
    return int((y * np.arange(1, len(y) + 1)).sum() % 1_000_003)


def d1_replicate(y, rng, E, B, X, fl, e_hat):
    """D1 on one plasmode replicate (paper 1 step 19): the outcome model is cross-fitted on the replicate with a
    seed drawn from the replicate's own generator after its outcome; ``fl`` and ``e_hat`` are the split's folds
    and cross-fitted propensity. Returns ``(log RR, P, mean GLM weight, events, checksum)``."""
    with threadpool_limits(limits=1):
        mu0, ws = crossfit_mu0(B, X, E, y, fl, int(rng.integers(2**31 - 1)))
        th, se = aipw_att(y, E, e_hat, mu0)
    return th, wald_p(th, se), float(np.mean(ws)), int(y.sum()), outcome_checksum(y)
