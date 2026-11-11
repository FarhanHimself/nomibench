# nomibench

**Calibrated, discoverability-matched benchmarks of gene nominations against genetic causal support.**

Does a set of nominated genes (from a co-expression or causal network, a curated target list, an omics screen)
carry more genetic support, such as SMR/HEIDI or colocalisation support, than its genes' *detectability*
predicts? Genes that are easy to instrument have strong, well-powered eQTLs and tend to be long, sit in
gene-dense regions and be well studied. Such genes are both nominated more often and "supported" more often, so
an unmatched comparison rewards discoverability rather than biology. `nomibench` estimates enrichment beyond
discoverability and calibrates the estimator before the contrast is read.

It is the released code of the benchmark in *[paper 1 citation, to be added]*.

## What it does

| Component | Method | Module |
|---|---|---|
| **G1** (primary) | Firth logistic regression of support on the nomination and a 3-df spline of the logit propensity score of nomination given discoverability. Penalised likelihood-ratio test and profile penalised-likelihood CIs (Firth 1993; Heinze & Schemper 2002) | `estimators`, `firth`, `propensity` |
| G1b, G1c | Propensity-quintile Mantel-Haenszel (Robins-Breslow-Greenland variance); observed/expected with an exact Poisson-binomial P | `estimators`, `tables` |
| W1, W2 | Within GWAS loci: conditional Firth logistic regression (Heinze & Puhr 2010), exact by enumeration or the Gail-Lubin-Rubinstein recursion; Mantel-Haenszel by locus | `conditional`, `estimators` |
| D1 | Doubly robust: cross-fitted AIPW for the ATT risk ratio with a GLM + gradient-boosting super learner, median over 50 fold splits (Chernozhukov et al. 2018) | `aipw` |
| Calibration | Plasmode simulation (Franklin et al. 2014) with three nulls (fitted, discoverability ×2, locus-clustered) and power scenarios. **Estimand targeting**: each scenario's exposure effect is solved so that G1's population estimand equals the target | `plasmode` |
| Negative controls | Gene sets drawn from the unexposed genes with probability proportional to the nomination's propensity odds, each re-analysed on the real outcome; registered pass band 3.5–6.5% at P < 0.05 | `negative_control` |
| Dependence | Genomic-block bootstrap (TSS windows, 2/5/10 Mb) | `bootstrap` |
| Readings | Pre-specified decision rules: "doubling excluded" vs "inconclusive"; a four-way rule; a positive-control gate; validity and band rules | `decisions` |

Calibration and negative controls never read the nominated genes' outcomes, so their verdicts can be fixed before
the contrast they govern.

## Install

```bash
pip install nomibench            # once released on PyPI
# or from a clone
pip install .
pip install ".[parquet]"         # optional: read .parquet gene tables on the command line
```

Python ≥ 3.10; NumPy, SciPy, pandas, statsmodels, patsy, scikit-learn and threadpoolctl.

## Quick start

The input is a table with one row per **testable** gene, meaning a gene that has an instrument in the QTL resource:

| column | meaning |
|---|---|
| `Y` | 0/1 genetic support (e.g. SMR q < 0.05 and HEIDI P > 0.05) |
| `E` | 0/1 nomination |
| discoverability covariates | e.g. log top-eQTL \|z\|, log10 eQTL sample size, log10 cis SNPs, log10 gene length, gene density |
| optional `interval`, `ld` | GWAS-locus interval id (empty outside loci); log10(distance to the index variant + 1 kb) |

```python
import nomibench as nb

cov = ["x_z", "x_N", "x_snp", "x_len", "x_den", "x_nd"]

# 1. Calibrate G1 for this nomination (simulated outcomes; nominated genes' outcomes are not read)
cal = nb.calibrate(genes, outcome="Y", exposure="E", covariates=cov, linear=["x_nd"],
                   generator_covariates=["x_z", "x_len", "x_snp"], locus="interval", locus_distance="ld",
                   seed=20260927, workers=4)
print(cal.calibrated)                 # type I error <= 0.065 under N0 and N1

# 2. Matched negative control on the real outcome (nominated genes' outcomes are not read)
sets, nc = nb.matched_controls(genes, "Y", "E", cov, linear=["x_nd"], n_sets=1000, seed=4, workers=4)
print(nc.reject, nc.passes)

# 3. The contrast
res = nb.evaluate(genes, "Y", "E", cov, linear=["x_nd"], locus="interval", locus_distance="ld")
print(res)
print(nb.decisions.network_rule(res.query("estimator == 'G1'").hi.iloc[0]))
```

Command line: `nomibench evaluate|calibrate|controls|d1 --data genes.csv --outcome Y --exposure E --covariates ...`
(`nomibench --help`).

With `workers > 1`, call the package from inside an `if __name__ == "__main__":` block, as `examples/quickstart.py`
does. Worker processes re-import the calling script under the spawn and forkserver start methods: the default on
Windows and macOS, and on Linux from Python 3.14. Results do not depend on the number of workers.

Every function that draws random numbers needs a seed, so every run can be repeated exactly.

## Validation

Every estimator is checked against an independent implementation or a known answer (`tests/`; details and
tolerances in `docs/VALIDATION.md`):

- **Firth**: R `logistf`'s results for its two standard examples, as recorded in firthlogist's tests (coefficients and profile intervals, to 5e-7); R `brglm2` (to 1e-8) and `firthlogist` on synthetic designs, including separation, and on G1's actual spline design; an independent quasi-Newton maximisation for restricted fits, LR statistics and a profile bound.
- **Conditional likelihood**: R `survival::clogit(method = "exact")`; enumeration against the recursion; analytic against numerical penalty gradients.
- **Mantel-Haenszel**: R `mantelhaen.test` and statsmodels. **Fisher**: R `fisher.test`. **Poisson-binomial**: brute force.
- **Plasmode**: estimand targeting to 1e-8; every replicate estimator equals the direct estimator on the same simulated outcome; G1's type I error on synthetic designs.
- **D1**: recovers a known ATT risk ratio; with the true nuisances, AIPW is unbiased. **Block bootstrap**: with one gene per block, its SE agrees with G1's model SE.

On paper 1's private inputs, the package regenerated the registered step 19–20 numbers it was checked against, in
the registered environment and in a newer one (`repro/paper1`, `docs/VALIDATION.md`):
- the ten registered estimates and every reported interval and P value;
- the first 50 replicates of every frozen plasmode file, from the registered seeds;
- the first 100 matched control sets of each control;
- the estimand-targeting constants and D1;
- after the specification, the amendment A10 outputs (E2-design plasmode, E2 matched control, block bootstrap).

## Convergence

`firth_fit` reports `converged`, `n_iter` and the largest modified score, and warns when the iteration cap binds.
The default cap is 1,000. Paper 1's registered analyses used 100 (`nomibench.registered.MAXIT`). At that cap
one registered fit stopped early; its log odds ratio differs from the converged value by 3.4e-8.

Under complete separation, a profile bound far from the estimate can be numerically unstable, because fitted
probabilities reach 1e-25. Treat such a bound as unbounded.

## Reproducing paper 1

`repro/paper1/` holds the reproduction script and instructions for the steps that need MetaBrain data. MetaBrain
results cannot be redistributed, so you need your own copy under its terms. No data ship with this package.

## Citation

See `CITATION.cff`. Please cite the archived release (Zenodo DOI) and the paper.

## Licence

BSD-3-Clause.
