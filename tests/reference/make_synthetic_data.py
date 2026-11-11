"""Synthetic datasets for the reference tests (all values simulated here; no real genes).

Run once; the CSVs in tests/data are committed. Re-running reproduces them exactly (fixed seeds).
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def firth_sets():
    rng = np.random.default_rng(20261006)
    sets = {}
    # 1. moderate: n = 400, three covariates, ~25% events
    n = 400
    X = rng.normal(size=(n, 3))
    eta = -1.2 + X @ np.array([0.8, -0.5, 0.3])
    sets["moderate"] = pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(y=rng.binomial(1, 1 / (1 + np.exp(-eta))))
    # 2. quasi-separation: n = 60, a binary covariate with no events in one level
    n = 60
    b = rng.binomial(1, 0.3, n)
    c = rng.normal(size=n)
    y = np.where(b == 1, rng.binomial(1, 0.9, n), rng.binomial(1, 0.25, n))
    y[(b == 0) & (c > 1.2)] = 0
    sets["quasi_separation"] = pd.DataFrame({"b": b, "c": c, "y": y})
    # 3. rare events with a G1-like design: exposure + covariate, ~2% events, n = 3000
    n = 3000
    z = rng.normal(size=n)
    e = rng.binomial(1, 1 / (1 + np.exp(-(-2 + 0.9 * z))))
    eta = -4.3 + 0.9 * z + 0.4 * e
    sets["rare_events"] = pd.DataFrame({"e": e, "z": z, "z2": z ** 2 / 3, "y": rng.binomial(1, 1 / (1 + np.exp(-eta)))})
    # 4. complete separation on one covariate, n = 40
    n = 40
    u = np.sort(rng.normal(size=n))
    v = rng.normal(size=n)
    y = (u > 0.2).astype(int)
    sets["separation"] = pd.DataFrame({"u": u, "v": v, "y": y})
    return sets


def conditional_set():
    """Stratified data: 25 strata of 6-40 units, 1-6 events each; two covariates."""
    rng = np.random.default_rng(20261007)
    rows = []
    for s in range(25):
        n = int(rng.integers(6, 41))
        x1 = rng.binomial(1, 0.3, n).astype(float)
        x2 = rng.normal(size=n)
        a = rng.normal(0, 1)
        p = 1 / (1 + np.exp(-(-2.0 + a + 0.7 * x1 + 0.4 * x2)))
        y = rng.binomial(1, p)
        if y.sum() == 0:
            y[int(rng.integers(n))] = 1
        rows.append(pd.DataFrame({"stratum": s, "x1": x1, "x2": x2, "y": y}))
    return pd.concat(rows, ignore_index=True)


def g1_design_set():
    """A realistic G1 design: rare outcome, confounded exposure, and the actual design [1, E, s(logit PS)] with
    the propensity score and spline basis built by nomibench (statsmodels and patsy), exported so that R brglm2
    and firthlogist can fit the identical matrix. Columns: y, e, lps, c0..c3 (the design)."""
    from nomibench.estimators import g1_design
    from nomibench.propensity import fit_ps
    rng = np.random.default_rng(20261009)
    n = 2500
    d = pd.DataFrame({"x_z": rng.normal(size=n), "x_len": rng.normal(size=n), "x_nd": rng.integers(1, 6, n).astype(float)})
    d["e"] = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 0.8 * d.x_z + 0.3 * d.x_len + 0.1 * d.x_nd)))).astype(float)
    eta = -3.6 + 0.8 * d.x_z + 0.2 * d.x_len + 0.35 * d.e
    d["y"] = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    ps = fit_ps(d, "e", ["x_z", "x_len", "x_nd"], linear=("x_nd",))
    X = g1_design(ps.logit, d.e.values)
    out = pd.DataFrame({"y": d.y, "e": d.e, "lps": ps.logit})
    for j in range(X.shape[1]):
        out[f"c{j}"] = X[:, j]
    return out


def stratified_tables():
    """2 x 2 x K counts: exposed events a, exposed non-events b, unexposed events c, unexposed non-events d."""
    rng = np.random.default_rng(20261008)
    K = 6
    return pd.DataFrame({"a": rng.integers(0, 12, K), "b": rng.integers(5, 60, K), "c": rng.integers(1, 15, K),
                         "d": rng.integers(20, 120, K)})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k, d in firth_sets().items():
        d.to_csv(os.path.join(OUT, f"firth_{k}.csv"), index=False, float_format="%.17g")
    conditional_set().to_csv(os.path.join(OUT, "conditional.csv"), index=False, float_format="%.17g")
    stratified_tables().to_csv(os.path.join(OUT, "mh_tables.csv"), index=False)
    g1_design_set().to_csv(os.path.join(OUT, "g1_design.csv"), index=False, float_format="%.17g")
    print("written to", os.path.abspath(OUT))
