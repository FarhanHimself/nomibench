"""Each plasmode estimator equals the corresponding direct estimator on the same simulated replicate; seeding,
summaries (equivalence and profile rows, failed fits) and the validity rule."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from conftest import make_genes
from nomibench import decisions
from nomibench import plasmode as pm
from nomibench.estimators import g1, g1_design, g1b, g1c, h_design, h_interaction, naive, w1
from nomibench.propensity import balance, fit_ps

COV = ["x_z", "x_len", "x_nd"]


@pytest.fixture(scope="module")
def setup():
    d = make_genes(n=4000, seed=41)
    E = d.E.values
    inl = d.L.values
    iid = np.full(len(d), -1)
    iid[inl] = pd.factorize(d.interval[inl])[0]
    ps = fit_ps(d, "E", COV, linear=("x_nd",))
    _, psq = balance(d, "E", COV, ps.ps)
    XG1 = g1_design(ps.logit, E)
    X_gen = np.c_[np.ones(len(d)), d.x_z, d.x_len, d.L.astype(float), d.ld]
    gam = pm.fit_generator(X_gen, np.where(E == 0, d.Y, np.nan), E)
    eta0, eta1, shift = pm.null_scenarios(X_gen, gam, inl, doubled=(1, 2))
    scen = pm.build_scenarios(XG1, eta0, eta1, shift, E, inl, n_null=3, n_power=3, power_ors=(2.0,))
    des = pm.PlasmodeDesign(E=E, X_g1=XG1, in_locus=inl, locus_id=iid, ps_quintile=np.asarray(psq),
                            X_oe=np.c_[np.ones(len(d)), d[COV].values], X_h=h_design(ps.logit, E, inl),
                            X_w1=np.c_[E, d.ld.values, ps.logit][inl])
    return d, ps, iid, scen, des


@pytest.mark.parametrize("k", [0, 2, 3])          # N0, N2 (locus random intercept), P-GW OR 2
def test_replicate_equals_direct_estimators(setup, k):
    d, ps, iid, scen, des = setup
    sc = scen[k]
    row = pm.replicate(sc, 0, np.random.SeedSequence(77), des, estimators=tuple(pm.ESTIMATORS))
    y, _ = pm.simulate(sc.eta, des.in_locus, des.locus_id, sc.csd, sc.shift, np.random.SeedSequence(77))
    E, inl = des.E, des.in_locus
    assert row["ysum"] == y.sum() and row["yexp"] == y[E == 1].sum()
    r = g1(y, E, ps.logit, profile_levels=(0.95,), wald_levels=())
    assert row["g1_logor"] == pytest.approx(r.log_or, abs=1e-12)
    assert row["g1_p"] == pytest.approx(r.pvalue, rel=1e-10)
    assert row["g1_se"] == pytest.approx(r.se, rel=1e-12)
    assert (row["g1_prof_lo"], row["g1_prof_hi"]) == pytest.approx(r.profile[0.95], abs=1e-9)
    # the equivalence event is stored as the event itself: 1 if the Wald 95% upper bound is below OR 2
    assert row["equiv_logor"] == pytest.approx(r.log_or, abs=1e-12)
    assert row["upper_lt_margin"] == float(r.log_or + norm.ppf(0.975) * r.se < np.log(2))
    fr = naive(y, E)                 # replicates store the sample (cross-product) OR, as registered
    a, b, c, d_ = y[E == 1].sum(), (1 - y[E == 1]).sum(), y[E == 0].sum(), (1 - y[E == 0]).sum()
    assert row["fisher_logor"] == pytest.approx(np.log(a * d_ / (b * c)), rel=1e-12)
    assert row["fisher_p"] == pytest.approx(fr.pvalue, rel=1e-12)
    o, _, _, p = g1b(y, E, des.ps_quintile)
    assert row["g1b_logor"] == pytest.approx(np.log(o), rel=1e-12) and row["g1b_p"] == pytest.approx(p, rel=1e-12)
    oe, p2, _, _ = g1c(y, E, des.X_oe)
    assert row["g1c_logoe"] == pytest.approx(np.log(oe), rel=1e-12) and row["g1c_p"] == pytest.approx(p2, rel=1e-12)
    hr = h_interaction(y, E, inl, ps.logit)
    assert row["h_logor"] == pytest.approx(np.log(hr[0]), abs=1e-12) and row["h_p"] == pytest.approx(hr[3], rel=1e-10)
    wr = w1(y[inl], E[inl], iid[inl], np.c_[d.ld.values[inl], ps.logit[inl]], profile=False)
    assert row["w1_logor"] == pytest.approx(wr.log_or, abs=1e-9) and row["w1_p"] == pytest.approx(wr.pvalue, rel=1e-8)


def test_failed_estimator_gives_nan_and_counts_as_non_rejection(setup):
    d, ps, iid, scen, des = setup
    bare = pm.PlasmodeDesign(E=des.E, X_g1=des.X_g1, in_locus=des.in_locus, locus_id=des.locus_id)   # no PS quintiles
    R = pm.run(bare, scen[:1], seeds=3, estimators=("g1", "g1b"))
    assert R.g1b_logor.isna().all() and R.g1b_p.isna().all() and R.g1_p.notna().all()
    s = pm.summarise(R).set_index("estimator").loc["g1b"]
    assert s.reject == 0 and s.failed == 1 and s.verdict == "FAILED FITS"


def test_classify_log_matches_the_four_way_rule():
    rng = np.random.default_rng(1)
    lo, hi = rng.normal(0, 0.6, 400), rng.normal(0.7, 0.6, 400)
    lo[:5] = np.nan
    name = {"enriched": "enriched", "modest": "modest enrichment", "no_material": "no material enrichment",
            "inconclusive": "inconclusive", "failed": "failed"}
    got = pm.classify_log(lo, hi)
    want = [decisions.four_way_rule(np.exp(a), np.exp(b)) for a, b in zip(lo, hi)]
    assert [name[g] for g in got] == want


def test_seeding_is_reproducible_and_positional():
    sc = [pm.Scenario("N0", np.zeros(3), 4), pm.Scenario("N1", np.zeros(3), 2)]
    a = pm.seeds_from_root(11, sc)
    b = pm.seeds_from_root(11, sc)

    def state(m, k):
        return [s.generate_state(2).tolist() for s in m[k]]

    assert state(a, "N1") == state(b, "N1")
    root = np.random.SeedSequence(11)
    assert state(pm.seeds_from_root(root, sc), "N0") == state(pm.seeds_from_root(root, sc), "N0")   # reuse-safe
    sc2 = [pm.Scenario("N0", np.zeros(3), 9), pm.Scenario("N1", np.zeros(3), 2)]
    c = pm.seeds_from_root(11, sc2)
    assert state(c, "N0")[:4] == state(a, "N0") and state(c, "N1") == state(a, "N1")   # replicate counts don't matter
    with pytest.raises(ValueError):
        pm.seeds_from_root(None, sc)


def test_summary_decision_rows():
    n = 10
    R = pd.DataFrame({"scenario": "N0", "rep": range(n), "g1_logor": 0.0, "g1_se": 0.3, "g1_p": 0.5,
                      "equiv_logor": 0.0, "upper_lt_margin": [1, 1, 0, 0, 0, 0, 0, 0, np.nan, np.nan],
                      "g1_prof_lo": -0.5, "g1_prof_hi": [0.5, 0.6, 0.8, 0.9, 1, 1, 1, 1, 1, np.nan]})
    S = pm.summarise(R, targets={"N0": 0.0}).set_index("estimator")
    eq = S.loc["g1_wald_upper_lt_margin"]
    assert eq.reject == pytest.approx(0.2) and eq.failed == pytest.approx(0.2)     # a failed fit is not an exclusion
    pr = S.loc["g1_profile_upper_lt_margin"]
    assert pr.reject == pytest.approx(0.2) and pr.failed == pytest.approx(0.1) and pr.coverage == pytest.approx(0.9)
    assert S.loc["g1_profile_upper_lt_margin_failed_as_excl", "reject"] == pytest.approx(0.3)  # ...except here


def test_validity_rule_needs_its_scenarios():
    R = pd.DataFrame({"scenario": ["N0"] * 20 + ["N1"] * 20, "rep": list(range(20)) * 2, "g1_logor": 0.0,
                      "g1_se": 0.3, "g1_p": np.r_[np.full(19, 0.5), [np.nan], np.full(20, 0.5)]})
    S = pm.summarise(R)
    assert pm.validity(S, "g1", ("N0", "N1"))
    assert not pm.validity(S, "g1", ("N0", "N1"), max_failed=0.01)       # 5% failed fits under N0
    with pytest.raises(ValueError):
        pm.validity(S, "g1", ("N0", "N2"))
