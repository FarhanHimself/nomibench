"""Propensity score, balance and the enrichment estimators on synthetic genes."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import nomibench as nb
from conftest import make_genes
from nomibench.estimators import g1, g1_design, g1b, g1c, h_design, h_interaction, naive, unadjusted_firth, w1, w2
from nomibench.propensity import balance, fit_ps, spline_formula

COV = ["x_z", "x_len", "x_nd"]


def test_spline_formula_rules(genes):
    f = spline_formula(genes, COV, linear=("x_nd",))
    assert "cr(x_z, df=3, constraints='center')" in f and "cr(x_len" in f and " x_nd" in f
    flat = genes.assign(k=np.r_[np.zeros(len(genes) - 5), np.ones(5)])        # 1/3 and 2/3 quantiles coincide
    assert spline_formula(flat, ["k"]) == "k"


def test_fit_ps_c_statistic_and_balance(genes):
    ps = fit_ps(genes, "E", COV, linear=("x_nd",))
    assert ps.converged
    assert ps.c_statistic == pytest.approx(roc_auc_score(genes.E, ps.ps), abs=1e-12)
    np.testing.assert_allclose(ps.logit, np.log(ps.ps / (1 - ps.ps)))
    bal, psq = balance(genes, "E", COV, ps.ps)
    assert bal.smd_raw.abs().max() > 0.3                     # the exposure is confounded by discoverability
    assert bal.smd_psq.abs().max() < 0.1                     # quintile weighting removes most of it
    assert set(np.unique(psq)) == set(range(5))


def test_rank_deficient_ps_raises(genes):
    with pytest.raises(ValueError):
        fit_ps(genes.assign(dup=genes.x_nd), "E", ["x_nd", "dup"], linear=("x_nd", "dup"))


def test_g1_recovers_null_and_effect():
    # discoverability confounds the naive comparison; G1 removes it
    d0 = make_genes(n=12000, seed=11, effect=0.0)
    ps = fit_ps(d0, "E", COV, linear=("x_nd",))
    r0 = g1(d0.Y.values, d0.E.values, ps.logit, profile_levels=(0.95,), wald_levels=(0.95, 0.975))
    lo, hi = r0.profile_or()
    assert lo < 1 < hi
    assert naive(d0.Y.values, d0.E.values).odds_ratio > r0.odds_ratio      # naive is inflated
    d1 = make_genes(n=12000, seed=12, effect=np.log(2.5))
    ps1 = fit_ps(d1, "E", COV, linear=("x_nd",))
    r1 = g1(d1.Y.values, d1.E.values, ps1.logit)
    assert r1.profile_or()[0] > 1 and 1.4 < r1.odds_ratio < 4.5
    # Wald 97.5% interval is wider than Wald 95%; both are centred on the estimate on the log scale
    w95, w975 = r0.wald[0.95], r0.wald[0.975]
    assert w975[0] < w95[0] < r0.log_or < w95[1] < w975[1]
    assert r0.converged and r0.n == len(d0) and r0.exposed == int(d0.E.sum())


def test_secondary_estimators_agree_roughly(genes):
    y, e = genes.Y.values, genes.E.values
    ps = fit_ps(genes, "E", COV, linear=("x_nd",))
    _, psq = balance(genes, "E", COV, ps.ps)
    r = g1(y, e, ps.logit)
    o_b, lo_b, hi_b, p_b = g1b(y, e, np.asarray(psq))
    oe, p_c, n_obs, n_exp = g1c(y, e, np.c_[np.ones(len(genes)), genes[COV].values])
    assert n_obs == int(y[e == 1].sum()) and n_exp > 0
    for v in (o_b, oe):
        assert abs(np.log(v) - r.log_or) < 0.5
    inl = genes.L.values
    iid = genes.interval.fillna(-1).astype(int).values
    o_h, lo_h, hi_h, p_h = h_interaction(y, e, inl, ps.logit)
    assert lo_h < o_h < hi_h
    wr = w1(y[inl], e[inl], iid[inl], np.c_[genes.ld.values[inl], ps.logit[inl]])
    assert wr is not None and wr.informative_strata > 0 and np.isfinite(wr.log_or)
    assert np.isfinite(w2(y[inl], e[inl], iid[inl])[0])
    o_u, lo_u, hi_u, p_u = unadjusted_firth(y, e)
    assert lo_u < o_u < hi_u


def test_designs_have_expected_columns(genes):
    lps = fit_ps(genes, "E", COV).logit
    X = g1_design(lps, genes.E.values)
    assert X.shape[1] == 5 and np.allclose(X[:, 0], 1) and np.array_equal(X[:, 1], genes.E.values)
    H = h_design(lps, genes.E.values, genes.L.values)
    assert H.shape[1] == 7 and np.array_equal(H[:, 3], genes.E.values * genes.L.values)


def test_evaluate_one_call(genes):
    res = nb.evaluate(genes, "Y", "E", COV, linear=("x_nd",), locus="interval", locus_distance="ld")
    assert {"naive (Fisher)", "G1"} <= set(res.estimator)
    assert res.attrs["exposed"] == int(genes.E.sum()) and res.attrs["max_abs_smd_after"] < 0.1
    assert res.estimator.str.startswith("W1").any() and res.estimator.str.startswith("H ").any()
    with pytest.raises(ValueError):
        nb.evaluate(genes.assign(Y=np.where(genes.index == 0, np.nan, genes.Y)), "Y", "E", COV)
