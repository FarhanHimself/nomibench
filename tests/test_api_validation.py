"""Entry points refuse bad input and unreproducible calls, and report estimators they cannot compute."""
import subprocess
import sys

import numpy as np
import pytest

import nomibench as nb
from conftest import make_genes
from nomibench import aipw, bootstrap, decisions, negative_control
from nomibench.benchmark import check_table, locus_arrays

COV = ["x_z", "x_len", "x_nd"]


@pytest.fixture(scope="module")
def small():
    return make_genes(n=1500, seed=3)


@pytest.mark.parametrize("change, msg", [
    (lambda d: d.assign(E=d.E.where(d.index != 0, np.nan)), "exposure"),
    (lambda d: d.assign(E=d.E.where(d.index != 0, 2.0)), "exposure"),
    (lambda d: d.assign(E=0.0), "no exposed"),
    (lambda d: d.assign(Y=d.Y.where(d.index != 5, np.nan)), "outcome"),
    (lambda d: d.assign(x_z=d.x_z.where(d.index != 0, np.inf)), "x_z"),
    (lambda d: d.drop(columns="x_len"), "x_len"),
])
def test_check_table_rejects_bad_tables(small, change, msg):
    with pytest.raises(ValueError, match=msg):
        check_table(change(small), "Y", "E", COV)


def test_exposed_outcomes_may_be_missing_where_they_are_never_read(small):
    d = small.assign(Y=np.where(small.E == 1, np.nan, small.Y))
    check_table(d, "Y", "E", COV, outcome_complete=False)
    with pytest.raises(ValueError):
        check_table(d, "Y", "E", COV)
    with pytest.raises(ValueError):
        nb.evaluate(d, "Y", "E", COV, linear=("x_nd",))
    nb.matched_controls(d, "Y", "E", COV, ("x_nd",), n_sets=2, seed=1)          # runs


def test_reproducibility_requires_a_seed(small):
    with pytest.raises(ValueError, match="seed"):
        nb.calibrate(small, "Y", "E", COV, ["x_z"], linear=("x_nd",), n_null=2, nulls=("N0",), power_ors=())
    with pytest.raises(ValueError, match="seed"):
        nb.matched_controls(small, "Y", "E", COV, ("x_nd",), n_sets=2)
    with pytest.raises(ValueError, match="seed"):
        negative_control.run(small, "Y", "E", COV, ("x_nd",), n_sets=2)
    with pytest.raises(ValueError, match="seed"):
        bootstrap.run(small, "Y", ["E"], COV, ("x_nd",), n_reps=2)
    with pytest.raises(ValueError, match="seeds"):
        aipw.d1(small, "Y", "E", COV, ("x_nd",), splits=2)


def test_calibrate_needs_locus_columns_for_locus_estimators(small):
    for kw in (dict(nulls=("N0", "N1", "N2")), dict(nulls=("N0", "N1"), estimators=("g1", "h")),
               dict(nulls=("N0", "N1"), estimators=("g1", "w1"))):
        with pytest.raises(ValueError, match="locus"):
            nb.calibrate(small, "Y", "E", COV, ["x_z"], linear=("x_nd",), n_null=2, power_ors=(), seed=1, **kw)


def test_calibrated_is_none_unless_both_nulls_ran(small):
    c = nb.calibrate(small, "Y", "E", COV, ["x_z"], linear=("x_nd",), n_null=3, nulls=("N0",), power_ors=(), seed=1)
    assert c.calibrated is None and c.generator_converged
    c = nb.calibrate(small, "Y", "E", COV, ["x_z"], linear=("x_nd",), n_null=3, nulls=("N0", "N1"), power_ors=(),
                     seed=1, estimators=("naive",))
    assert c.calibrated is None
    c = nb.calibrate(small, "Y", "E", COV, ["x_z"], linear=("x_nd",), n_null=3, nulls=("N0", "N1"), power_ors=(),
                     seed=1)
    assert c.calibrated in (True, False)


def test_locus_identifiers_of_any_type(small):
    d = small.assign(interval=small.interval.map(lambda v: np.nan if np.isnan(v) else f"chr{int(v)}:locus"))
    inl, iid, ld = locus_arrays(d, "interval", "ld")
    inl0, iid0, _ = locus_arrays(small, "interval", "ld")
    np.testing.assert_array_equal(inl, inl0)
    assert len(set(zip(iid, iid0))) == len(set(iid)) == len(set(iid0))        # the same grouping of genes
    assert (iid[~inl] == -1).all() and (ld[~inl] == 0).all()
    ids = small.interval[inl0].astype(int)
    if set(ids) == set(range(ids.max() + 1)):                                  # integer ids 0..K-1 keep their values
        np.testing.assert_array_equal(iid0[inl0], ids.values)
    with pytest.warns(UserWarning, match="empty"):
        locus_arrays(small.assign(interval=small.interval.fillna(0)), "interval")
    with pytest.raises(ValueError, match="in-locus"):
        locus_arrays(small.assign(ld=np.nan), "interval", "ld")


def test_evaluate_reports_estimators_it_cannot_compute(small):
    d = small.assign(E=np.where(small.L, 0.0, small.E))                 # no exposed gene inside a locus
    res = nb.evaluate(d, "Y", "E", COV, linear=("x_nd",), locus="interval", locus_distance="ld")
    h = res[res.estimator.str.startswith("H ")]
    w = res[res.estimator.str.startswith("W1")]
    assert "not estimable" in h.estimator.iloc[0] and h.OR.isna().all() and not h.converged.any()
    assert "not estimable" in w.estimator.iloc[0] and w.OR.isna().all()
    assert np.isfinite(res[res.estimator == "G1"].OR).all()


def test_decision_rules_for_the_gate():
    assert decisions.gate_rule(2.53, power=0.999) == "pass"
    assert decisions.gate_rule(0.99, power=0.999) == "fail"
    assert decisions.gate_rule(np.nan, power=0.999) == "fail"
    assert decisions.gate_rule(2.53, power=0.75) == "uninformative"                 # A5: power at OR 2.7 < 0.80
    assert decisions.gate_rule(2.53, power=0.95, calibrated=False) == "uninformative"
    cr = decisions.coprimary_reading
    assert cr(0.5, 1.8, True, True, "FAIL") == "not interpretable as absence (gate failed)"   # any case
    assert cr(0.5, 1.8, True, True, "uninformative") == "no material enrichment"           # as registered
    with pytest.raises(ValueError):
        cr(0.5, 1.8, True, True, "failed")
    assert decisions.network_rule(np.nan) == "inconclusive"


def test_cli_reports_bad_input_without_a_traceback(tmp_path, small):
    p = tmp_path / "genes.csv"
    small.drop(columns="x_len").to_csv(p, index=False)
    r = subprocess.run([sys.executable, "-m", "nomibench.cli", "evaluate", "--data", str(p), "--outcome", "Y",
                        "--exposure", "E", "--covariates", "x_z,x_len"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "x_len" in r.stderr and "Traceback" not in r.stderr


@pytest.mark.filterwarnings("ignore")              # statsmodels warns on the separated fit
def test_fit_ps_c_statistic_and_separation(small):
    from nomibench.propensity import c_stat, fit_ps
    assert np.isnan(c_stat(np.full(5, 0.3), np.ones(5)))
    sep = small.assign(E=(small.x_z > 0.5).astype(float))
    with pytest.raises(ValueError, match="separates"):
        fit_ps(sep, "E", ["x_z"], linear=("x_z",))
