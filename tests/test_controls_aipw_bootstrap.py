"""Matched negative controls, D1 (AIPW) and the block bootstrap on synthetic genes; the decision rules."""
import numpy as np
import pandas as pd
import pytest

import nomibench as nb
from conftest import make_genes
from nomibench import aipw, bootstrap, decisions, negative_control

COV = ["x_z", "x_len", "x_nd"]


def test_draw_sets_follow_propensity_odds():
    rng = np.random.default_rng(0)
    ps = np.r_[np.full(500, 0.05), np.full(500, 0.5)]
    elig = np.ones(1000, bool)
    sets = negative_control.draw_sets(ps, elig, 20, 400, rng)
    assert all(len(np.unique(s)) == 20 for s in sets)
    share_high = np.mean([np.mean(s >= 500) for s in sets])
    assert 0.85 < share_high < 0.98                       # odds 1 vs 0.0526: ~95% expected, without replacement


def test_negative_control_never_reads_exposed_outcomes(genes):
    d1 = genes.copy()
    d2 = genes.assign(Y=np.where(genes.E == 1, 1 - genes.Y, genes.Y))
    r1 = negative_control.run(d1, "Y", "E", COV, ("x_nd",), n_sets=5, seed=4)
    r2 = negative_control.run(d2, "Y", "E", COV, ("x_nd",), n_sets=5, seed=4)
    assert r1.equals(r2)


@pytest.mark.slow
def test_negative_control_rejection_near_nominal():
    d = make_genes(n=5000, seed=31)
    R, s = nb.matched_controls(d, "Y", "E", COV, ("x_nd",), n_sets=300, seed=8, workers=2)
    assert 0.02 <= s.reject <= 0.09 and s.failed == 0


def test_aipw_with_true_nuisances_is_unbiased():
    rng = np.random.default_rng(5)
    n = 200_000
    x = rng.normal(size=n)
    e_true = 1 / (1 + np.exp(-(-1 + 0.8 * x)))
    E = rng.binomial(1, e_true).astype(float)
    mu0 = 1 / (1 + np.exp(-(-3 + 0.7 * x)))
    rr = 1.8
    Y = rng.binomial(1, np.where(E == 1, np.minimum(rr * mu0, 1), mu0)).astype(float)
    th, se = aipw.aipw_att(Y, E, e_true, mu0)
    assert abs(th - np.log(rr)) < 3 * se and se < 0.05


def test_d1_runs_deterministically(genes):
    a = aipw.d1(genes, "Y", "E", COV, ("x_nd",), splits=2, entropy=[1, 2])
    b = aipw.d1(genes, "Y", "E", COV, ("x_nd",), splits=2, entropy=[1, 2], workers=2)
    assert a == b and a.splits_used == 2 and a.lo < a.rr < a.hi
    assert aipw.split_seeds([1, 2], 2) == aipw.split_seeds([1, 2], 2)


def test_super_learner_probabilities(genes):
    B = aipw.basis(genes, COV, ("x_nd",))
    X = genes[COV].values
    p, w = aipw.super_learner(B, X, genes.E.values, B, X, seed=3)
    assert np.all((p > 0) & (p < 1)) and 0 <= w <= 1


def test_block_bootstrap(genes):
    blocks = bootstrap.make_blocks(genes.chr.values, genes.tss.values, 5e6)
    assert sum(len(b) for b in blocks) == len(genes)
    B = bootstrap.run(genes, "Y", ["E"], COV, ("x_nd",), chrom="chr", tss="tss", n_reps=6, seed=3)
    B2 = bootstrap.run(genes, "Y", ["E"], COV, ("x_nd",), chrom="chr", tss="tss", n_reps=6, seed=3, workers=2)
    assert B.equals(B2) and B.lnOR_E.notna().all()
    s = bootstrap.summarise(B, "E")
    assert s["pct_lo"] < s["pct_hi"]


@pytest.mark.slow
def test_d1_recovers_a_known_att_risk_ratio():
    # Y(1) = Bernoulli(rr * mu0(X)) for the exposed, so the ATT risk ratio is exactly rr; mu0 and e are nonlinear
    # in X, so the learners must find them
    rng = np.random.default_rng(12)
    n, rr = 20000, 1.7
    d = pd.DataFrame({"x_z": rng.normal(size=n), "x_len": rng.normal(size=n), "x_nd": rng.integers(1, 6, n).astype(float)})
    e = 1 / (1 + np.exp(-(-1.6 + 0.7 * d.x_z + 0.4 * d.x_len ** 2 - 0.3)))
    d["E"] = rng.binomial(1, e).astype(float)
    mu0 = 1 / (1 + np.exp(-(-3.3 + 0.8 * d.x_z + 0.3 * np.abs(d.x_len))))
    d["Y"] = rng.binomial(1, np.where(d.E == 1, rr * mu0, mu0)).astype(float)
    r = aipw.d1(d, "Y", "E", COV, ("x_nd",), splits=3, entropy=4, workers=2)
    assert abs(r.log_rr - np.log(rr)) < 3 * r.se and r.se < 0.1
    assert r.lo < rr < r.hi


@pytest.mark.slow
def test_block_bootstrap_se_matches_model_se_for_independent_genes():
    # one gene per block (unique TSS, 1-bp blocks): the block bootstrap is the ordinary bootstrap, and for
    # independent genes its SE of G1's log OR should agree with the model-based SE
    d = make_genes(n=3000, seed=17, effect=0.5).assign(chr=1, tss=np.arange(3000) * 10.0)
    from nomibench.estimators import g1
    from nomibench.propensity import fit_ps
    r = g1(d.Y.values, d.E.values, fit_ps(d, "E", COV, ("x_nd",)).logit, profile_levels=())
    B = bootstrap.run(d, "Y", ["E"], COV, ("x_nd",), chrom="chr", tss="tss", block_size=1, n_reps=200, seed=9,
                      workers=2)
    assert len(bootstrap.make_blocks(d.chr, d.tss, 1)) == 3000
    s = bootstrap.summarise(B, "E")
    assert s["failed"] == 0
    assert 0.8 < s["boot_se"] / r.se < 1.25                      # Monte Carlo error of the SE at 200 replicates ~ 5%


def test_decision_rules():
    assert decisions.network_rule(1.41) == "doubling excluded" and decisions.network_rule(2.24) == "inconclusive"
    assert decisions.network_rule(2.0) == "inconclusive"
    f = decisions.four_way_rule
    assert f(1.09, 5.44) == "enriched" and f(1.05, 1.9) == "modest enrichment"
    assert f(0.6, 1.9) == "no material enrichment" and f(0.33, 2.88) == "inconclusive"
    assert f(np.nan, 2.0) == "failed"
    assert decisions.gate_rule(2.53, power=0.999) == "pass" and decisions.gate_rule(0.99, power=0.999) == "fail"
    assert decisions.validity_rule(0.047, 0.0445) and not decisions.validity_rule(0.07, 0.04)
    assert decisions.band_rule(0.041) and not decisions.band_rule(0.034) and not decisions.band_rule(0.069)
    assert decisions.robustness_rule(-0.17) and not decisions.robustness_rule(0.2)
    cr = decisions.coprimary_reading
    assert cr(0.33, 2.88, True, True, "pass") == "inconclusive"
    assert cr(0.9, 2.6, True, False, "pass") == "not interpreted"
    assert cr(0.5, 1.8, True, True, "fail") == "not interpretable as absence (gate failed)"
