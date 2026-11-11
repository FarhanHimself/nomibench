"""Paper 1's registered settings, so that its analyses can be re-run exactly with this package.

Paper 1: a calibrated, discoverability-matched benchmark of Alzheimer's disease gene nominations against SMR/HEIDI
support (MetaBrain cortex-EUR cis-eQTL x the 2026 consensus AD GWAS). Column names refer to the U_gw covariate
table built by ``repro/paper1/build_covariates.py``, a port of paper 1's step-19 script ``19a`` (the table is
MetaBrain-derived and not distributed).
"""
TIER_P = ("x_N", "x_snp", "x_nd", "x_len", "x_den", "x_z")
"""Discoverability covariates: log10 eQTL sample size, log10 cis SNPs tested, contributing datasets,
log10 gene length, protein-coding gene density within 1 Mb, log top-eQTL |z|. Order matters for D1's trees."""

TIER_P_LINEAR = ("x_nd",)
"""Covariates entered linearly in the propensity model (the rest are 3-df cubic regression splines)."""

GENERATOR_COLUMNS = ("x_z", "x_len", "x_snp", "L", "ld")
"""Plasmode generator terms after the intercept: log top-eQTL |z|, log10 length, log10 SNPs, in-locus flag,
in-locus x log10(distance to the nearest index variant + 1 kb)."""

GENERATOR_DOUBLED = (1, 2, 3)
"""Generator coefficients doubled under N1 (the three discoverability terms)."""

N2_SD = 1.0

MAXIT = 100
"""Firth iteration cap used by every registered analysis (the package default is 1,000)."""

SEED_STEP19 = 20260927
"""Step 19: SeedSequence root of the E1 plasmode; children spawned in turn for N0, N1, N2 (2,000 each) and
P-GW OR 1.5, 2.0, 2.7 (1,000 each)."""
SEED_E4 = (20260927, 4)
"""Step 19: default_rng seed of the E4 matched sets."""
SEED_D1_E1 = (20260927, 1)
"""Step 19: D1 split-seed entropy for E1; E1w uses (20260927, 1, 2) and E2 (20260927, 1, 3)."""
SEED_A5 = 20260928
"""Step 20: root of the E5 seed tree: 7 exposure children (order of E5_SETS), 7 scenario children each
(N0, N1, N2, P-GW OR 1.5, 2.0, 2.7, NC), one child per replicate. D1 entropy (20260928, 1, k), k = 1..7."""
SEED_A10 = 20261004
"""Step 25 (A10): root spawned as N0, N1, N2, P1.5, P2.0, NC_E2, BOOT5, BOOT2, BOOT10."""

E5_SETS = ("E5a", "E5c", "E5g", "E5b", "E5e", "E5d", "E5c_naive")
E5_SCENARIOS = ("N0", "N1", "N2", "P-GW_OR1.5", "P-GW_OR2.0", "P-GW_OR2.7", "NC")
E5_FULL = ("E5a", "E5c", "E5g")          # 2,000 null / 1,000 power replicates; the others N0 and N1 x 1,000
E5_COPRIMARY = ("E5a", "E5c")
E5_GATE = "E5g"

Z_STEP19 = 1.959964
"""Step 19 used this rounded 97.5% normal quantile for the Wald equivalence check and D1's interval;
step 20 onwards used scipy's exact value."""

MARGIN = 2.0
NC_BAND = (0.035, 0.065)
VALIDITY_THRESHOLD = 0.065
ROBUSTNESS_DLN = 0.2

FROZEN_SHA256 = {
    "ugw_covariates.csv": "f12c36baf7706d1df355de2e525922146286a45cf771875eb3e4527218b84974",
    "smr_gw_all.csv": "295dddce10c3421c6ca34ab285c99bcce65625d32eec5d95587fce1d55f80428",
    "e5_exposure_sets_A5.tsv": "31f4b17432105574b9c994e6b3a14d2858a15c7a6935328e74658a6d98abb06a",
}
"""Frozen inputs of paper 1 (private; listed so a reader with access can check their copies)."""
