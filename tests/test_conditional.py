"""Exact conditional likelihood: the two evaluators agree, the penalty gradient is right, and the unpenalised fit
matches R survival::clogit(method = "exact")."""
import os

import numpy as np
import pandas as pd
import pytest

import nomibench.conditional as cl
from conftest import DATA


@pytest.fixture(scope="module")
def strat():
    d = pd.read_csv(os.path.join(DATA, "conditional.csv"))
    return d[["x1", "x2"]].values.astype(float), d.y.values.astype(float), d.stratum.values


def _with_enum_max(v, X, y, sid):
    old = cl.ENUM_MAX
    cl.ENUM_MAX = v
    try:
        return cl.make_strata(X, y, sid)
    finally:
        cl.ENUM_MAX = old


def test_dynamic_programme_equals_enumeration(strat):
    X, y, sid = strat
    # keep strata small enough to enumerate, then force the recursion on the same strata
    small = np.isin(sid, [s for s in np.unique(sid) if (sid == s).sum() <= 16])
    se = _with_enum_max(10 ** 9, X[small], y[small], sid[small])
    sd = _with_enum_max(0, X[small], y[small], sid[small])
    assert all(s[0] == "E" for s in se) and sd.dp is not None and all(s[0] == "D" for s in sd)
    for beta in (np.zeros(2), np.array([0.7, -0.4]), np.array([-1.5, 2.0])):
        a = cl._cond_all(se, beta, third=True)
        b = cl._cond_all(sd, beta, third=True)
        assert a[0] == pytest.approx(b[0], abs=1e-10)
        for u, v in zip(a[1:], b[1:]):
            np.testing.assert_allclose(u, v, atol=1e-9)


def test_penalty_gradient_matches_finite_differences(strat):
    X, y, sid = strat
    st = cl.make_strata(X, y, sid)
    beta = np.array([0.5, 0.3])
    _, U, I, M3 = cl._cond_all(st, beta, third=True)
    g = U + 0.5 * np.einsum("ij,ijk->k", np.linalg.pinv(I), M3)
    h = 1e-6
    fd = np.array([(cl._cfl_pen(st, beta + h * e) - cl._cfl_pen(st, beta - h * e)) / (2 * h) for e in np.eye(2)])
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-7)


def test_unpenalised_fit_matches_r_clogit(strat, r_ref):
    X, y, sid = strat
    st = cl.make_strata(X, y, sid)
    f = cl.cfl_fit(st, 2, penalised=False)
    assert f.converged
    np.testing.assert_allclose(f.beta, r_ref["clogit_coef"], atol=1e-9)
    assert f.loglik == pytest.approx(r_ref["clogit_loglik"], abs=1e-9)


def test_penalised_fit_lrt_and_profile(strat):
    X, y, sid = strat
    st = cl.make_strata(X, y, sid)
    f = cl.cfl_fit(st, 2)
    t = cl.cfl_lrt(st, 2, 0)
    lo, hi = cl.cfl_profile_ci(st, 2, 0, f.beta, f.loglik)
    assert f.converged and t.converged
    assert lo < f.beta[0] < hi and 0 <= t.pvalue <= 1
    # Firth's penalty shrinks towards zero relative to the unpenalised conditional MLE
    fu = cl.cfl_fit(st, 2, penalised=False)
    assert abs(f.beta[0]) < abs(fu.beta[0])
    b, stat, p, ll = t                                               # legacy unpacking
    assert np.allclose(b, f.beta)


def test_uninformative_strata_are_dropped():
    X = np.ones((6, 1))
    y = np.array([0, 0, 0, 1, 1, 1.0])
    sid = np.array([0, 0, 0, 1, 1, 1])
    assert len(cl.make_strata(X, y, sid)) == 0
