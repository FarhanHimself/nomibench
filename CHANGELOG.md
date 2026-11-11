# Changelog

## 1.0.0 (2026-10-06)

First release, accompanying paper 1. The estimators are the registered analysis code, ported without changing
their arithmetic, and reproduce the registered numbers (`docs/VALIDATION.md`).

Changes from the registered scripts:

**Convergence and failures**
- `firth_fit` reports `converged`, `n_iter` and the largest modified score, and warns when its iteration cap
  binds. The default cap is 1,000; the registered analyses used 100 (`nomibench.registered.MAXIT`).
- `firth_profile_ci` judges convergence only at the restricted fit nearest each bound, and reports `saturated`
  when a fitted probability there is below 1e-15 or above 1 - 1e-15 (complete separation; treat the bound as
  unbounded). An unbracketed bound is NaN, as before.
- `cfl_fit` reports convergence (cap unchanged at 200).
- Every plasmode estimator is wrapped so that a failed fit returns NaN for that replicate; NaN counts as a
  non-rejection, as registered. Summaries report the failure rate, mark a scenario with half or more failed
  fits "FAILED FITS", and give the profile false-exclusion rate both ways (failure as non-exclusion and as
  exclusion).
- `fit_ps` raises if the propensity model separates the exposure; its c-statistic is NaN, not an error, when a
  group is empty.

**Decisions**
- The plasmode's equivalence check stores the event itself (Wald upper bound below the margin). The registered
  step-19 code stored it as a 0/1 pseudo-P, which its report read the wrong way round (paper 1, step-28 erratum).
- `gate_rule(lo, power, calibrated)` applies the registered evaluability condition (calibrated, and power at
  OR 2.7 of at least 0.80) and returns "uninformative" otherwise. `coprimary_reading` accepts the gate in any
  case and rejects unknown values, so a typo cannot silently skip the downgrade.

**Inputs and reproducibility**
- One-call entry points (`evaluate`, `calibrate`, `matched_controls`), result objects and a command line.
- Gene tables are validated (0/1 coding, completeness, finite covariates) before any fit; the command line prints
  a one-line error instead of a traceback.
- Every simulation, set draw, bootstrap and fold split requires a seed. A `SeedSequence` passed as a root is
  copied, so reusing it reproduces the same streams.
- `calibrate` refuses N2, H and W1 without locus columns, reports `calibrated = None` unless both N0 and N1 were
  simulated with G1, and also requires at most 5% failed G1 fits under each null (`max_failed`; not part of
  the registered rule). It builds every requested estimator's inputs (G1b quintiles, G1c and H designs, W1
  strata). Locus identifiers of any type are factorised in sorted order, so integer ids 0..K-1 keep their values.
- `evaluate` reports H or W1 as "not estimable", with NaN, when the data cannot identify them.
- The `linear` argument defaults to no linear covariates everywhere; paper 1's choice is
  `nomibench.registered.TIER_P_LINEAR`.
- The plasmode's Wald equivalence check uses the exact normal quantile by default (`PlasmodeDesign.z_equiv`);
  step 19 used 1.959964 (`nomibench.registered.Z_STEP19`).
- The bootstrap's percentile interval uses exactly the 0.025 and 0.975 quantiles at level 0.95, as registered.
- D1's gradient boosting runs single-threaded, so results do not depend on the core count.
