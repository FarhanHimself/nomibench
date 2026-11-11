"""Firth logistic regression against independent implementations.

References (tests/reference): firthlogist 0.5.0 (follows R logistf), R brglm2 0.9.2 (bias-reduced GLM, which for
logistic regression is Firth's estimator), and an independent quasi-Newton maximisation written here.
"""
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

from conftest import load_firth_set
from nomibench import ConvergenceWarning
from nomibench.firth import firth_fit, firth_lrt, firth_profile_ci, wald_ci, wald_se

SETS = ["moderate", "quasi_separation", "rare_events", "separation"]


@pytest.mark.parametrize("name", SETS)
def test_coefficients_match_firthlogist_and_brglm2(name, firth_ref, r_ref):
    X, y = load_firth_set(name)
    f = firth_fit(X, y)
    assert f.converged
    tol = 1e-6 if name == "separation" else 1e-8            # flat penalised likelihood under separation
    np.testing.assert_allclose(f.beta, firth_ref[name]["coef"], atol=tol)
    np.testing.assert_allclose(f.beta, r_ref[f"brglm2_{name}"], atol=tol)
    assert f.loglik == pytest.approx(firth_ref[name]["loglik"], abs=1e-8)
    assert f.max_abs_score < 1e-6


@pytest.mark.parametrize("name", SETS)
def test_profile_ci_and_lrt_match_firthlogist(name, firth_ref):
    X, y = load_firth_set(name)
    f = firth_fit(X, y)
    ref = firth_ref[name]
    for j in range(X.shape[1]):
        ci = firth_profile_ci(X, y, j, f.beta, f.loglik, span=20.0)
        lo_ref, hi_ref = ref["ci"][j]
        assert ci.lo == pytest.approx(lo_ref, abs=5e-6)
        if name == "separation" and j == 1:
            # The separated coefficient's upper bound lies where fitted probabilities reach 1e-25: the penalised
            # likelihood is numerically flat there and implementations disagree (nomibench 24.4, firthlogist 25.0,
            # an independent BFGS about 22). Treat such a bound as "very large", not as a number.
            assert ci.hi > 15
            continue
        assert ci.hi == pytest.approx(hi_ref, abs=5e-6)
        assert ci.converged
    # P-values are not compared with firthlogist: its restricted fit computes leverages from the reduced design
    # (columns masked), so it does not maximise the full-design penalised likelihood that logistf and nomibench
    # profile (Heinze & Schemper 2002); its LR statistics run about 1e-3 higher. The statistic is checked against
    # an independent exact maximisation in test_against_independent_maximisation.


def _independent_firth(X, y, fixed=None):
    """Penalised log-likelihood maximised by BFGS with an analytic gradient; leverages from a thin QR of
    W^1/2 X (no explicit inverse). Written independently of nomibench.firth."""
    p = X.shape[1]
    fixed = fixed or {}
    free = [j for j in range(p) if j not in fixed]
    base = np.zeros(p)
    for j, v in fixed.items():
        base[j] = v

    def f(bf):
        b = base.copy()
        b[free] = bf
        eta = X @ b
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        sign, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
        Q, _ = np.linalg.qr(X * np.sqrt(W)[:, None], mode="reduced")
        h = np.einsum("ij,ij->i", Q, Q)
        g = X.T @ (y - mu + h * (0.5 - mu))
        return -(ll + 0.5 * logdet), -g[free]

    x = np.zeros(len(free))
    for _ in range(5):
        r = minimize(f, x, jac=True, method="BFGS", options=dict(gtol=1e-10, maxiter=50000))
        x = r.x
    b = base.copy()
    b[free] = x
    return b, -r.fun


@pytest.mark.parametrize("name", ["moderate", "rare_events", "quasi_separation"])
def test_against_independent_maximisation(name):
    X, y = load_firth_set(name)
    f = firth_fit(X, y)
    b, lp = _independent_firth(X, y)
    np.testing.assert_allclose(f.beta, b, atol=1e-6)
    assert f.loglik == pytest.approx(lp, abs=1e-9)
    # restricted fit (the penalty keeps the full design) and the likelihood-ratio statistic
    f0 = firth_fit(X, y, fixed={1: 0.0})
    b0, lp0 = _independent_firth(X, y, fixed={1: 0.0})
    np.testing.assert_allclose(f0.beta, b0, atol=1e-6)
    t = firth_lrt(X, y, 1)
    assert t.stat == pytest.approx(2 * (lp - lp0), abs=1e-7)
    # profile bound by independent root-finding
    crit = chi2.ppf(0.95, 1) / 2

    def g(v):
        return lp - _independent_firth(X, y, fixed={1: v})[1] - crit

    hi = brentq(g, b[1] + 1e-6, b[1] + 8, xtol=1e-9)
    assert firth_profile_ci(X, y, 1, f.beta, f.loglik).hi == pytest.approx(hi, abs=5e-6)


def test_convergence_flag_and_warning():
    X, y = load_firth_set("rare_events")
    with pytest.warns(ConvergenceWarning):
        f = firth_fit(X, y, maxit=2)
    assert not f.converged and f.n_iter == 2 and f.max_abs_score > 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = firth_fit(X, y)
    assert f.converged and f.n_iter < 50


def test_separation_gives_finite_estimates():
    X, y = load_firth_set("separation")
    f = firth_fit(X, y)
    assert np.all(np.isfinite(f.beta)) and f.converged


def test_legacy_unpacking_and_wald():
    X, y = load_firth_set("moderate")
    b, lp = firth_fit(X, y)
    beta, stat, p = firth_lrt(X, y, 1)
    lo, hi = firth_profile_ci(X, y, 1)
    se = wald_se(X, b, 1)
    wl, wh = wald_ci(b[1], se, 0.95)
    assert lo < b[1] < hi and wl < b[1] < wh
    assert np.allclose(beta, b) and 0 <= p <= 1
    # profile and Wald intervals agree closely with 400 observations and 104 events
    assert abs(lo - wl) < 0.05 and abs(hi - wh) < 0.05


def test_fixed_everything_returns_immediately():
    X, y = load_firth_set("moderate")
    f = firth_fit(X, y, fixed={j: 0.0 for j in range(X.shape[1])})
    assert f.converged and f.n_iter == 0 and np.all(f.beta == 0)
