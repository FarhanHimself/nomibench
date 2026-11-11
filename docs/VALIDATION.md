# Validation of nomibench 1.0.0

Two layers of evidence:

1. **The test suite** (`tests/`, shipped): every estimator against an independent implementation or a known
   answer, on synthetic data and on two published example datasets.
2. **The reproduction of paper 1** (`repro/paper1/`, private inputs): the package regenerates the registered
   numbers from the registered seeds. Only aggregate differences are reported here.

## 1. Test suite

83 tests (79 fast, 4 slow). The full suite passed in three environments:

| Environment | Python | numpy | scipy | pandas | statsmodels | patsy | scikit-learn |
|---|---|---|---|---|---|---|---|
| Development (editable install) | 3.11.15 | 2.4.4 | 1.17.1 | 2.3.3 | 0.15.0 | 1.0.3 | 1.8.0 |
| Paper 1's step-20 environment (wheel installed; tests run from the sdist) | 3.11.15 | 2.4.4 | 1.17.1 | 3.0.2 | 0.15.0 | 1.0.3 | 1.8.0 |
| Clean environment, latest releases (wheel installed; tests run from the sdist) | 3.11.15 | 2.4.6 | 1.17.1 | 3.0.6 | 0.15.0 | 1.0.3 | 1.9.1 |

Continuous integration (`.github/workflows/tests.yml`) runs the fast suite on Linux, Windows and macOS with
Python 3.10–3.13, the slow suite on Linux, and the linter.

### Agreement with independent implementations

"Asserted" is the tolerance the test enforces; "observed" is the largest difference measured when the reference
values were generated.

| Component | Reference | Asserted | Observed |
|---|---|---|---|
| Firth coefficients and profile 95% intervals, `sex2` and `endometrial` | R logistf's results for its two standard examples, as recorded in the tests of firthlogist 0.5.0 (which assert agreement at rtol 1e-5). The data are in `tests/data`, with their provenance | 5e-7 absolute | 2.0e-7 (logistf prints 8 decimals and stops at its own tolerance) |
| Firth coefficients, four synthetic designs (moderate, quasi-separation, rare events, complete separation) | R 4.3.3, brglm2 0.9.2 (`type = "AS_mean"`, epsilon 1e-12) | 1e-8; 1e-6 under separation | 1.9e-10 |
| Firth coefficients, G1's actual design (intercept, exposure, 3-df spline of the logit propensity score; 2,500 genes, 3.8% events) | brglm2 0.9.2 on the identical exported matrix | 1e-8 | 3.7e-10 |
| The same, and the exposure's profile 95% interval | firthlogist 0.5.0 (`fit_intercept=False`) | 5e-7; interval 5e-6 | 1.2e-7; interval 1.5e-7 |
| Firth coefficients and profile intervals, synthetic designs | firthlogist 0.5.0 | coefficients 1e-8 (separation 1e-6); intervals 5e-6 | 6.9e-8 (separation) |
| Restricted fits, LR statistic, one profile bound | independent quasi-Newton (BFGS) maximisation of the penalised likelihood | coefficients 1e-6; statistic 1e-7; bound 5e-6 | — |
| Conditional likelihood: unpenalised fit | R survival 3.5.8 `clogit(method = "exact")` | 1e-9 | — |
| Conditional likelihood: recursion and penalty gradient | enumeration; central finite differences | 1e-9; rtol 1e-6 | — |
| Mantel–Haenszel OR and RBG interval | R `mantelhaen.test`; statsmodels `StratifiedTable` | OR rtol 1e-12; interval rtol 1e-9 | — |
| Fisher conditional MLE, exact interval and P | R `fisher.test` | rtol 2e-4 (R finds the estimate and interval with `uniroot` at tolerance eps^0.25); P rtol 1e-9 | — |
| Poisson-binomial distribution and tails | brute-force enumeration | 1e-14 absolute | — |

firthlogist's restricted fits mask the fixed column in the hat matrix, so they are not the exact maximisers of
the full-design penalised likelihood; its LR statistics therefore differ from logistf's by about 1e-3 and are
not used as a reference. The independent BFGS maximisation is used instead.

### Known answers and internal consistency

- **Plasmode.** Estimand targeting reaches its target to 1e-8. Each replicate estimator (G1, its profile
  interval, the equivalence event, naive Fisher, G1b, G1c, H, W1) equals the direct estimator on the same
  simulated outcome, under N0, N2 and a power scenario. Seeds are reproducible, reuse-safe and positional.
  Summaries count failures as non-rejections and report the cautious false-exclusion row. *Slow:* G1's type I
  error lies between 0.02 and 0.085 (5% ± about 3 Monte Carlo SE) under N0, N1 and N2 on a synthetic design,
  400 replicates each, with mean estimates within 0.05 of the target.
- **Negative controls.** Sets follow the propensity odds; the exposed genes' outcomes are never read. *Slow:*
  rejection between 0.02 and 0.09 over 300 sets, with no failed fit.
- **D1.** With the true nuisances, AIPW recovers a known risk ratio. *Slow:* D1 recovers a known ATT risk ratio of
  1.7 within 3 SE when the propensity and outcome models are nonlinear.
- **Block bootstrap.** Deterministic, and independent of the number of workers. *Slow:* with one gene per block
  (the ordinary bootstrap), the ratio of its SE of G1's log OR to the model-based SE lies between 0.8 and 1.25
  (200 replicates).
- **Entry points.** Bad tables, missing seeds and estimators the data cannot identify are refused or reported;
  the command line reports bad input without a traceback; worker pools run under the spawn start method.
- **Golden regression.** Fixed synthetic data and seeds; stored outputs of every entry point (D1 as in §3).
- **Governance.** No Ensembl gene ID or private path in any shipped file. The only gene symbols in the release
  are the registered E3 input list (12 public genes) in `repro/paper1/build_covariates.py` and the MHC/APOE
  region name; no gene-level result ships.

## 2. Reproduction of paper 1

**Inputs.** The frozen U_gw covariate table (`f12c36ba…`), SMR table (`295dddce…`) and exposure sets
(`31f4b174…`), all private (MetaBrain-derived), and the frozen aggregate outputs of steps 19, 20 and 25.
The script ran with `--require-frozen`, so all three hashes matched.

**Environments.** A: paper 1's step-20 environment, with the built wheel installed. B: the development
environment; of the listed libraries it differs from A only in pandas (2.3.3 instead of 3.0.2). The criteria R1–R9 were fixed in the
step-29 specification before the first run; R10 was added afterwards and is labelled post-spec.

| | Checked | Tolerance | A | B |
|---|---|---|---|---|
| R1 | the ten registered G1 estimates (maxit 100) | 1e-8 | 8.1e-14 | 1.1e-16 |
| R2 | 83 rows of the step-19 and step-20 result tables: estimates, intervals and P values of naive Fisher, unadjusted Firth, G1 (profile and Wald, 95% and 97.5%), G1b, G1c, H, W1, W2, sensitivity tiers, E3, E4 | estimate 1e-6 rel.; bounds 1e-5 log; P 1e-5 rel. | 6.8e-13 | 2.5e-12 |
| R3 | generator coefficients and N2 shift (7 sets) | 1e-8 | 4.4e-16 | 4.4e-16 |
| R3 | the estimand achieved by the package's β\*, recomputed; β\*, achieved and as-written estimand ORs against the A6 run; as-written ORs against the A5 run | 1e-8 | ≤ 2.0e-13 | ≤ 2.0e-13 |
| R4 | E1 plasmode, from the step-19 seeds: 6 scenarios × 50 replicates × 7 tests | 1e-8 | 4.4e-16 | 1.3e-9 |
| R4 | E5 plasmode, from the A5 seed tree: 40 files × 50 replicates; simulated event counts | identical | identical | identical |
| R4 | the same: G1 and Fisher ln OR, P, SE | 1e-8 | 2.2e-16 | 9.8e-9 |
| R4b | D1 on 10 E1 N0 replicates: outcome checksums and event counts (ln RR informational) | identical | identical (7.6e-17) | identical (7.6e-17) |
| R5 | plasmode summaries recomputed from the frozen replicate files (gw19, A6 and as-written) | 1e-12 | 2.2e-16 | 2.2e-16 |
| R6 | matched control sets, first 100 of NC_E5a and NC_E5c: event counts | identical | identical | identical |
| R6 | the same: ln OR and P | 1e-8 | 2.2e-16 | **1.4e-8 (fails, NC_E5a)**; 3.5e-10 (NC_E5c) |
| R6 | E4 sets (laptop run; informational) | — | identical; 1.1e-16 | identical; 1.4e-10 |
| R7 | D1 for E5a on the real outcome, 50 splits: RR and bounds | 1e-6 log | 2.2e-16 | 2.2e-16 |
| R7b | D1 for E1 (laptop run; informational) | — | 0 | 0 |
| R8 | the decision rules on the registered estimates (14 readings, including gate evaluability) | identical | identical | identical |
| R9 | convergence flags: all ten converge at the default cap; at maxit 100 only E1 does not | as expected | as expected | as expected |
| R9 | E1's ln OR between maxit 100 and the default | 1e-7 | 3.4e-8 | 3.4e-8 |
| R10 | A10 E2-design plasmode: β\*, achieved and as-written estimands | 1e-8 | 4.4e-16 | 4.4e-16 |
| R10 | the same, 5 scenarios × 50 replicates: event counts; G1 ln OR, P, SE and profile bounds | identical; 1e-8 | identical; 3.7e-11 | identical; 3.4e-11 |
| R10 | E2-matched control, first 100 sets: event counts; ln OR and P | identical; 1e-8 | identical; 2.6e-10 | identical; 3.8e-11 |
| R10 | 5-Mb block bootstrap, first 50 replicates: gene and event counts; ln OR of E1 and E2 | identical; 1e-3 | identical; 3.4e-6 | identical; 3.4e-6 |

**Result.** In paper 1's own environment (A) every criterion passes: 28 checks, 4 informational. In B, 27 pass
and one fails: NC_E5a by 1.4e-8 against a tolerance of 1e-8.

**Why B fails one check.** One set of the 100 (set 28) differs by 1.4e-8 in ln OR and 6e-12 in P. Its Firth
fit converged in 16 iterations, stopping on the step criterion (1e-9); at that point its largest modified score
was 5.7e-7, so the estimate is determined only to about 1e-8. Under pandas 2.3.3 its propensity design differs
in the last bits, and the stopping point moves by that much. The registered value is reproduced to 2.2e-16 in
the registered environment, so this is environment drift, not a package error, and no registered number,
interval or reading changes. It is reported, not waived: under the specification's tolerance B fails R6.

**Other differences.** R4's 1.3e-9 and 9.8e-9 in B have the same cause as the R6 failure: the registered values
are reproduced to 2.2e-16 in A. The full-precision R1 values and the A10 outputs (R10) came from the step-25 run
of 3 October; differences from them of 1e-14 to 3e-10 are at the level of floating-point rounding, and their
exact source (environment or operation order) was not traced. In the block bootstrap, 71 of the 100 replicate-exposure values are identical; the others differ by up to
3.4e-6, identically in A and B. A resample repeats genes, so its logit propensity scores contain exact ties,
and G1's spline knots are placed on the unique values; a one-ulp split of a tie moves a knot. Percentile
intervals over 2,000 replicates are not affected at any reported precision.

## 3. Numerical limits users should know

- **Firth stopping rule.** Fits stop when the largest step is below 1e-9. In nearly flat designs the estimate is
  then reproducible across environments only to about 1e-8.
- **Iteration cap.** The registered analyses capped Firth at 100 iterations; E1 stopped there, 3.4e-8 from its
  converged ln OR. The package default is 1,000, and `converged` is reported.
- **Separation.** A profile bound far from the estimate is unstable under complete separation (`saturated`);
  treat it as unbounded.
- **D1.** Gradient-boosted trees are deterministic given the seed, but their splits change between
  scikit-learn releases. On the golden synthetic data, scikit-learn 1.9.1 instead of 1.8.0 moves D1's RR by
  0.4% (P 0.060 instead of 0.063). Paper 1's D1 values were computed with 1.8.0 and are reproduced exactly with
  it (R4b, R7, R7b); the golden test requires exact agreement only under the scikit-learn minor version that
  wrote it, and agreement within 5% otherwise.

## 4. The validated source

The reproduction ran against these source files (the tree hash covers `src/nomibench/*.py`; documentation and
metadata edits do not change it):

```
8428d06125b9b6e4d6d0f646d9a8a215d7a0d9741aba7fc5f706a0e3c9e0ca49
```

```bash
python - <<'EOF'
import hashlib, pathlib
h = hashlib.sha256()
for p in sorted(pathlib.Path("src/nomibench").glob("*.py")):
    h.update(p.name.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
print(h.hexdigest())
EOF
```
