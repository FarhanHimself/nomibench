"""Fisher, Mantel-Haenszel (RBG) and Poisson-binomial against R and brute force."""
import itertools
import os

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.contingency_tables import StratifiedTable

from conftest import DATA
from nomibench.tables import fisher, fisher_logor, mh_rbg, poibin_pmf, poibin_two_sided, strat_counts


def _expand(a, b, c, d):
    y = np.r_[np.ones(a), np.zeros(b), np.ones(c), np.zeros(d)]
    e = np.r_[np.ones(a + b), np.zeros(c + d)]
    return y, e


def test_fisher_matches_r(r_ref):
    for t, ref in zip(r_ref["fisher_tables"], r_ref["fisher"]):
        y, e = _expand(*t)
        r = fisher(y, e)
        # R's fisher.test finds the conditional MLE and CI with uniroot at tolerance eps^0.25 (about 1e-4)
        np.testing.assert_allclose([r.odds_ratio, r.lo, r.hi], ref[:3], rtol=2e-4, atol=1e-12)
        assert r.pvalue == pytest.approx(ref[3], rel=1e-9)


def test_fisher_logor_is_sample_odds_ratio():
    y, e = _expand(12, 500, 170, 10400)
    lo, p = fisher_logor(y, e)
    assert lo == pytest.approx(np.log(12 * 10400 / (500 * 170)))
    y, e = _expand(0, 30, 12, 200)
    assert np.isnan(fisher_logor(y, e)[0])


def test_mantel_haenszel_matches_r_and_statsmodels(r_ref):
    T = pd.read_csv(os.path.join(DATA, "mh_tables.csv"))
    o, lo, hi, p = mh_rbg(T.a, T.b, T.c, T.d)
    assert o == pytest.approx(r_ref["mh_or"], rel=1e-12)
    np.testing.assert_allclose([lo, hi], r_ref["mh_ci"], rtol=1e-9)
    st = StratifiedTable([np.array([[a, b], [c, d]]) for a, b, c, d in T[["a", "b", "c", "d"]].values])
    assert o == pytest.approx(st.oddsratio_pooled, rel=1e-12)
    np.testing.assert_allclose([lo, hi], st.oddsratio_pooled_confint(), rtol=1e-9)


def test_strat_counts_round_trip():
    T = pd.read_csv(os.path.join(DATA, "mh_tables.csv"))
    ys, es, ss = [], [], []
    for k, (a, b, c, d) in enumerate(T[["a", "b", "c", "d"]].values):
        y, e = _expand(a, b, c, d)
        ys.append(y)
        es.append(e)
        ss.append(np.full(len(y), k))
    got = strat_counts(np.concatenate(ys), np.concatenate(es), np.concatenate(ss))
    for g, col in zip(got, "abcd"):
        np.testing.assert_array_equal(g, T[col].values)


def test_poisson_binomial_against_enumeration():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.4, 11)
    exact = np.zeros(len(p) + 1)
    for bits in itertools.product([0, 1], repeat=len(p)):
        bits = np.array(bits)
        exact[bits.sum()] += np.prod(np.where(bits == 1, p, 1 - p))
    np.testing.assert_allclose(poibin_pmf(p), exact, atol=1e-15)
    for obs in range(len(p) + 1):
        two, upper = poibin_two_sided(p, obs)
        assert upper == pytest.approx(exact[obs:].sum(), abs=1e-14)
        assert two == pytest.approx(min(1.0, exact[exact <= exact[obs] * (1 + 1e-9)].sum()), abs=1e-14)


def test_mh_undefined_returns_nan():
    assert all(np.isnan(v) for v in mh_rbg([0, 0], [5, 6], [0, 0], [7, 8]))
