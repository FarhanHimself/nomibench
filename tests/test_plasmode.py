"""Plasmode: estimand targeting, determinism, summaries, calibration on synthetic genes."""
import numpy as np
import pytest

import nomibench as nb
from conftest import make_genes
from nomibench import plasmode as pm
from nomibench.estimators import g1_design
from nomibench.propensity import fit_ps

COV = ["x_z", "x_len", "x_nd"]


def _design(d):
    E = d.E.values
    inl = d.L.values
    ps = fit_ps(d, "E", COV, linear=("x_nd",))
    XG1 = g1_design(ps.logit, E)
    X_gen = np.c_[np.ones(len(d)), d.x_z, d.x_len, d.L.astype(float), d.ld]
    gam = pm.fit_generator(X_gen, np.where(E == 0, d.Y, np.nan), E)
    eta0, eta1, shift = pm.null_scenarios(X_gen, gam, inl, doubled=(1, 2))
    return E, inl, XG1, X_gen, gam, eta0, eta1, shift


@pytest.fixture(scope="module")
def design():
    return make_genes(n=6000, seed=21)


def test_generator_ignores_exposed_outcomes(design):
    E = design.E.values
    X_gen = np.c_[np.ones(len(design)), design.x_z]
    y1 = np.where(E == 0, design.Y, np.nan)
    y2 = np.where(E == 0, design.Y, 1.0 - design.Y)                 # exposed outcomes flipped
    np.testing.assert_array_equal(pm.fit_generator(X_gen, y1, E), pm.fit_generator(X_gen, y2, E))


def test_null_scenarios_preserve_event_rates(design):
    E, inl, XG1, X_gen, gam, eta0, eta1, shift = _design(design)
    p0 = 1 / (1 + np.exp(-eta0))
    p1 = 1 / (1 + np.exp(-eta1))
    assert p1.mean() == pytest.approx(p0.mean(), rel=1e-10)          # N1: mean rate kept
    pN2 = pm.marginal_p(eta0, inl, 1.0, shift)
    assert pN2[inl].mean() == pytest.approx(p0[inl].mean(), rel=1e-8)   # N2: in-locus marginal rate kept


def test_estimand_targeting_hits_targets(design):
    E, inl, XG1, X_gen, gam, eta0, eta1, shift = _design(design)
    for sc in pm.build_scenarios(XG1, eta0, eta1, shift, E, inl, n_null=10, n_power=10):
        th = pm.g1_target(XG1, pm.marginal_p(sc.eta, inl, sc.csd, sc.shift))
        assert th == pytest.approx(sc.target_log_or, abs=1e-8)


def test_untargeted_null_is_not_a_null_when_exposure_is_locus_enriched(design):
    # locus terms are in the generator but not in the propensity model; the exposure is enriched in loci,
    # so beta = 0 implies a non-null G1 estimand (paper 1, amendment A6)
    E, inl, XG1, X_gen, gam, eta0, eta1, shift = _design(design)
    assert inl[E == 1].mean() > inl[E == 0].mean()
    sc = pm.build_scenarios(XG1, eta0, eta1, shift, E, inl, n_null=10, nulls=("N0",), power_ors=(), targeting="none")[0]
    assert abs(np.log(sc.target_or_as_written)) > 0.02


def test_simulation_is_deterministic_and_worker_independent(design):
    E, inl, XG1, X_gen, gam, eta0, eta1, shift = _design(design)
    iid = design.interval.fillna(-1).astype(int).values
    scen = pm.build_scenarios(XG1, eta0, eta1, shift, E, inl, n_null=6, n_power=3)
    dsg = pm.PlasmodeDesign(E=E, X_g1=XG1, in_locus=inl, locus_id=iid)
    a = pm.run(dsg, scen, seeds=123, estimators=("g1", "naive"), workers=1)
    b = pm.run(dsg, scen, seeds=123, estimators=("g1", "naive"), workers=2)
    assert a.equals(b)
    y1, _ = pm.simulate(scen[2].eta, inl, iid, 1.0, shift, np.random.SeedSequence(5))
    y2, _ = pm.simulate(scen[2].eta, inl, iid, 1.0, shift, np.random.SeedSequence(5))
    np.testing.assert_array_equal(y1, y2)


def test_summary_and_validity_rule():
    import pandas as pd
    R = pd.DataFrame({"scenario": ["N0"] * 100 + ["N1"] * 100, "rep": list(range(100)) * 2,
                      "g1_logor": 0.0, "g1_se": 0.3, "g1_p": np.r_[np.full(5, 0.01), np.full(95, 0.5), np.full(8, 0.01), np.full(92, 0.5)],
                      "fisher_logor": 0.0, "fisher_p": 0.5})
    S = pm.summarise(R)
    g = S[S.estimator == "g1"].set_index("scenario")
    assert g.loc["N0", "reject"] == 0.05 and g.loc["N1", "reject"] == 0.08
    assert g.loc["N0", "verdict"] == "calibrated" and g.loc["N1", "verdict"] == "ANTI-CONSERVATIVE"
    assert not pm.validity(S)


@pytest.mark.slow
def test_g1_type_one_error_on_synthetic_design(design):
    cal = nb.calibrate(design, "Y", "E", COV, ["x_z", "x_len"], linear=("x_nd",), locus="interval",
                       locus_distance="ld", n_null=400, n_power=100, power_ors=(2.0,), seed=99, workers=2)
    s = cal.summary[cal.summary.estimator == "g1"].set_index("scenario")
    for n in ("N0", "N1", "N2"):
        assert 0.02 <= s.loc[n, "reject"] <= 0.085                    # 5% +/- 3 Monte Carlo SE at 400 replicates
        assert abs(s.loc[n, "mean_minus_target"]) < 0.05
    assert s.loc["P-GW_OR2.0", "reject"] > 0.3
