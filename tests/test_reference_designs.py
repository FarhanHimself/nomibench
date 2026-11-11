"""Firth estimates against R logistf's results for its standard examples, and against two independent
implementations on G1's actual design (intercept, exposure and the 3-df spline of the logit propensity score)."""
import os

import numpy as np
import pandas as pd
import pytest

from conftest import DATA
from nomibench.estimators import g1, g1_design
from nomibench.firth import firth_fit, firth_profile_ci

# R logistf results for its two standard examples, as recorded in the test suite of firthlogist 0.5.0
# (firthlogist/tests/test_firthlogist.py), which asserts agreement with them at rtol 1e-5. Order: intercept,
# then the covariates. Intervals are profile penalised-likelihood 95% intervals.
LOGISTF = {
    "sex2": dict(y="case", x=["age", "oc", "vic", "vicl", "vis", "dia"],
                 coef=[0.12025405, -1.10598131, -0.06881673, 2.26887464, -2.11140817, -0.78831694, 3.09601166],
                 ci=[[-0.8185591, 1.07315122], [-1.9737884, -0.30742514], [-0.9414363, 0.78920202],
                     [1.2730216, 3.43543273], [-3.2608611, -1.11773495], [-1.6080879, 0.01518468],
                     [0.7745682, 8.03029352]]),
    "endometrial": dict(y="HG", x=["NV", "PI", "EH"], coef=[3.77455951, 2.92927330, -0.03475175, -2.60416387],
                        ci=[[1.0825371, 7.20928050], [0.6097244, 7.85463171], [-0.1244587, 0.04045547],
                            [-4.3651832, -1.23272106]]),
}


@pytest.mark.parametrize("name", list(LOGISTF))
def test_matches_published_logistf_results(name):
    ref = LOGISTF[name]
    d = pd.read_csv(os.path.join(DATA, f"logistf_{name}.csv"))
    X = np.c_[np.ones(len(d)), d[ref["x"]].values.astype(float)]
    y = d[ref["y"]].values.astype(float)
    f = firth_fit(X, y)
    assert f.converged
    # logistf prints 8 decimals and stops at its own convergence tolerance: agreement is to about 2e-7
    np.testing.assert_allclose(f.beta, ref["coef"], rtol=0, atol=5e-7)
    for j, (lo, hi) in enumerate(ref["ci"]):
        ci = firth_profile_ci(X, y, j, f.beta, f.loglik)
        assert ci.converged and not ci.saturated
        assert ci.lo == pytest.approx(lo, abs=5e-7) and ci.hi == pytest.approx(hi, abs=5e-7)


@pytest.fixture(scope="module")
def g1_set():
    g = pd.read_csv(os.path.join(DATA, "g1_design.csv"))
    return g, g[[c for c in g.columns if c.startswith("c")]].values


def test_g1_design_is_the_exported_matrix(g1_set):
    g, X = g1_set
    np.testing.assert_allclose(g1_design(g.lps.values, g.e.values), X, rtol=0, atol=1e-12)


def test_g1_design_against_brglm2_and_firthlogist(g1_set, r_ref, firth_ref):
    g, X = g1_set
    f = firth_fit(X, g.y.values)
    assert f.converged
    np.testing.assert_allclose(f.beta, r_ref["brglm2_g1_design"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(f.beta, firth_ref["g1_design"]["coef"], rtol=0, atol=5e-7)
    r = g1(g.y.values, g.e.values, g.lps.values)
    assert r.log_or == pytest.approx(r_ref["brglm2_g1_design"][1], abs=1e-8)
    np.testing.assert_allclose(r.profile[0.95], firth_ref["g1_design"]["ci"][1], rtol=0, atol=5e-6)
