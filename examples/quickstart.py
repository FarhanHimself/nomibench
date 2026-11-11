"""A complete benchmark of one nomination source on simulated genes (runs in about a minute).

    python examples/quickstart.py

Discoverability (eQTL strength, gene length) drives both nomination and support, so the unadjusted comparison
shows enrichment where there is none. G1 removes it, and the calibration and the matched negative control check
the pipeline before the contrast is read.
"""
import numpy as np
import pandas as pd

import nomibench as nb


def main():
    rng = np.random.default_rng(2026)
    n = 8000
    genes = pd.DataFrame({"x_z": rng.normal(size=n), "x_len": rng.normal(size=n), "x_nd": rng.integers(1, 6, n)})
    genes["L"] = rng.random(n) < 0.05                                    # inside a GWAS locus interval
    genes["interval"] = np.where(genes.L, rng.integers(0, 60, n), np.nan)
    genes["ld"] = np.where(genes.L, np.log10(rng.uniform(1e3, 5e5, n) + 1e3), 0.0)
    genes["E"] = rng.binomial(1, 1 / (1 + np.exp(-(-2.2 + 0.8 * genes.x_z + 0.3 * genes.x_len))))
    genes["Y"] = rng.binomial(1, 1 / (1 + np.exp(-(-4.3 + 0.9 * genes.x_z + 2.2 * genes.L))))   # no true effect of E
    cov = ["x_z", "x_len", "x_nd"]

    # 1. calibration (the nominated genes' outcomes are not read)
    cal = nb.calibrate(genes, "Y", "E", cov, generator_covariates=["x_z", "x_len"], linear=["x_nd"], locus="interval",
                       locus_distance="ld", n_null=200, n_power=100, power_ors=(2.0,), seed=1, workers=2)
    print(cal.summary.query("estimator in ['g1', 'naive']")[["scenario", "estimator", "reject", "mc_se", "verdict"]])
    print("G1 calibrated:", cal.calibrated)

    # 2. matched negative control on the real outcome (the nominated genes' outcomes are not read)
    sets, nc = nb.matched_controls(genes, "Y", "E", cov, linear=["x_nd"], n_sets=200, seed=2, workers=2)
    print(f"negative control: {nc.reject:.3f} rejected at P < 0.05 (band 3.5-6.5%: {'pass' if nc.passes else 'fail'})")

    # 3. the contrast
    res = nb.evaluate(genes, "Y", "E", cov, linear=["x_nd"], locus="interval", locus_distance="ld")
    print(res.to_string(index=False))
    g1 = res[(res.estimator == "G1") & (res.interval == "profile 95.0%")].iloc[0]
    print("reading:", nb.decisions.network_rule(g1.hi))


if __name__ == "__main__":      # needed for worker processes on Windows and macOS
    main()
