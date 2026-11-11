# Methods implemented in nomibench

## The question and the estimand

The unit is a **testable gene**: a gene with an instrument in the QTL resource. Each gene carries:
- **Y**, a binary outcome: genetic causal support, such as SMR q < 0.05 with HEIDI P > 0.05;
- **E**, a binary exposure: whether a source nominated it;
- **X**, discoverability covariates: properties that make support easier to detect, whatever the gene's biology. These include eQTL strength, eQTL sample size, cis SNPs tested, contributing datasets, gene length and gene density.

The estimand is the odds ratio of support for nominated against other genes at equal discoverability. G1 estimates it by
conditioning on the propensity score e(X) = P(E = 1 | X). D1 targets the risk ratio among nominated genes (ATT).

Discoverability confounds the comparison because it raises both the chance of nomination and the chance of
support. Matching on it is not optional when the nomination source was itself built from eQTL-anchored data.

## Propensity score

- Logistic regression of E on X.
- Each covariate enters as a natural cubic regression spline with 3 df (patsy `cr(..., constraints='center')`). Covariates named as linear, or whose 1/3 and 2/3 quantiles coincide, enter linearly.
- Diagnostics:
  - the c-statistic;
  - standardised mean differences, before and after weighting by propensity-score quintile.

## Estimators

- **G1 (primary).**
  - Model: Firth logistic regression `Y ~ E + s(logit e(X))`, with a 3-df cubic regression spline.
  - Estimation: penalised maximum likelihood with Jeffreys' prior (Firth 1993). The bias reduction matters because support is rare (about 2% of genes in paper 1) and nominated sets can be small.
  - Test: the penalised likelihood-ratio test.
  - Intervals: profile penalised-likelihood intervals (Heinze & Schemper 2002). Restricted fits keep the penalty of the full design, as in R `logistf`.
  - Wald intervals use the expected information at the Firth estimate.
- **G1b.** Mantel-Haenszel common odds ratio across propensity-score quintiles, with the Robins-Breslow-Greenland variance.
- **G1c.** Observed against expected supported genes among the nominated.
  - The expectation comes from a Firth model of Y on X fitted to the unexposed genes only.
  - The P-value is exact Poisson-binomial.
- **H.** The exposure × in-locus interaction, added to the G1 model.
- **W1.** Within GWAS loci: conditional logistic regression with strata = locus intervals and Firth's penalty (Heinze & Puhr 2010).
  - The conditional likelihood is exact: by enumeration for small strata, and by the Gail-Lubin-Rubinstein recursion otherwise.
  - The recursion is extended here to the first three moments of the subset-sum statistic, which gives the analytic penalty gradient.
- **W2.** Mantel-Haenszel by locus interval.
- **D1.** Cross-fitted augmented inverse-probability weighting for the ATT risk ratio.
  - Folds: five, stratified on E.
  - Nuisances: e(X) and mu0(X) = P(Y = 1 | X, E = 0). Each is a two-member super learner: the spline GLM and gradient-boosted trees, combined with a convex weight chosen by log-loss on an 80/20 split.
  - Inference: influence-function SE.
  - Aggregation: 50 independent fold splits combined by the median rule (Chernozhukov et al. 2018).

## Calibration by plasmode simulation

The real X, E and locus structure are kept; only Y is simulated (Franklin et al. 2014).

**Generator.**
- A Firth logistic model of Y on [1, discoverability terms, in-locus flag L, L × log10 distance to the index variant].
- It is fitted to the **unexposed** genes, so nominated genes' outcomes are never read.

**Scenarios.**

| Scenario | Linear predictor |
|---|---|
| N0 | the fitted generator |
| N1 | discoverability coefficients doubled; intercept re-solved to keep the mean event rate |
| N2 | N0 plus a normal random intercept per locus interval (SD 1), shifted to keep the in-locus marginal rate (40-point Gauss-Hermite) |
| P-GW OR r | N0 plus log(r) × E |

**Estimand targeting.**
- The generator's locus terms are not in the propensity model. When nominated genes are enriched in loci, setting the exposure coefficient to 0 therefore does not give a null for G1's estimand.
- For each scenario, the package solves the exposure coefficient β\* at which G1's population estimand equals the scenario's target. The estimand is the exposure coefficient of the logistic projection of the scenario's marginal P(Y = 1) onto G1's design. Performance is then measured against the estimand's true value, as simulation studies require (Morris, White & Crowther 2019); the coefficient is set by iteration because a propensity-score estimand is not a parameter of the generator (Austin 2007).
- In paper 1, an untargeted generator would have labelled 4 of 7 external nomination sets miscalibrated, a positive control among them.

**Validity rule.** An estimator is calibrated for a design if its type I error at P < 0.05 is at most 0.065 under
both N0 and N1. Monte Carlo SEs are reported. A failed fit counts as a non-rejection, so `calibrate()` also
requires at most 5% failed G1 fits under each null (an addition to the registered rule; set `max_failed=None` to
apply the registered rule alone), and the summary marks a scenario with half or more failed fits "FAILED FITS".

**Replicate estimators.** Each replicate can be analysed with G1 (penalised LRT, Wald SE), its profile interval,
the Wald equivalence event (upper bound below the margin), G1b, G1c, H, W1 and the naive Fisher test. In
replicates the naive estimate is the log sample (cross-product) odds ratio, as registered; the real-data naive row
reports the conditional MLE. Both use Fisher's exact P.

**Decision operating characteristics.** For G1 the package also reports:
- P(lower bound > 1) and P(upper bound < margin);
- the shares of each four-way reading;
- for the profile interval, P(upper bound < margin) twice: counting a failed or unbracketed bound as not below
  the margin, and as below it (the cautious reading of a false exclusion), with the interval's coverage.

## Matched negative controls on the real outcome

- **Sets.** Random sets the size of the nominated set are drawn from the unexposed genes, without replacement. The probability of each gene is proportional to the nomination's propensity odds e/(1 − e).
- **Analysis.** Each set gets its own propensity score and G1 fit on the unexposed genes' real outcomes.
- **Pass band.** Under a calibrated pipeline about 5% of sets reject at P < 0.05. Paper 1's registered band is 3.5–6.5%.

## Genomic-block bootstrap

- **Blocks.** Genes are grouped into contiguous windows by TSS (block = chromosome and floor(TSS / size)).
- **Resampling.** Blocks are resampled with replacement, keeping the number of blocks.
- **Refitting.** Both the propensity score and G1 are refitted in every replicate. Intervals are percentile intervals.

## Decision rules (paper 1)

| Contrast | Rule |
|---|---|
| Network nominations (E1, E2) | "doubling excluded" if the profile 95% upper bound < 2; otherwise "inconclusive" (never "no enrichment") |
| Co-primaries (E5a, E5c), Wald 97.5% interval | lower > 1: "enriched" (upper ≥ 2) or "modest enrichment" (upper < 2); lower ≤ 1: "no material enrichment" (upper < 2) or "inconclusive" (upper ≥ 2) |
| Positive-control gate (E5g) | evaluated only if the control is calibrated and G1's plasmode power at OR 2.7 (rejection at P < 0.05) is ≥ 0.80, otherwise "uninformative"; when evaluated, pass if the Wald 95% lower bound > 1 |
| Co-primary reading | not interpreted unless calibrated and the matched control passes; a failed gate turns "no material enrichment" into "not interpretable as absence"; an uninformative gate leaves the reading unchanged, as registered, so its status is reported beside it |
| Robustness tier | passes if \|Δ ln OR\| < 0.2 |

A missing bound reads "inconclusive" under the network rule and "failed" under the four-way rule.

## References

Checked against PubMed on 6 Oct 2026, except Firth 1993, Gail 1981 and Chernozhukov 2018, which PubMed does not index.

- Firth D. Bias reduction of maximum likelihood estimates. *Biometrika* 1993;80:27–38.
- Heinze G, Schemper M. A solution to the problem of separation in logistic regression. *Stat Med* 2002;21:2409–2419. doi:10.1002/sim.1047
- Heinze G, Puhr R. Bias-reduced and separation-proof conditional logistic regression with small or sparse data sets. *Stat Med* 2010;29:770–777. doi:10.1002/sim.3794
- Gail MH, Lubin JH, Rubinstein LV. Likelihood calculations for matched case-control studies and survival studies with tied death times. *Biometrika* 1981;68:703–707. doi:10.1093/biomet/68.3.703
- Robins J, Breslow N, Greenland S. Estimators of the Mantel-Haenszel variance consistent in both sparse data and large-strata limiting models. *Biometrics* 1986;42:311–323. PMID 3741973
- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. *Stat Med* 2019;38:2074–2102. doi:10.1002/sim.8086
- Austin PC. The performance of different propensity score methods for estimating marginal odds ratios. *Stat Med* 2007;26:3078–3094. doi:10.1002/sim.2781
- Franklin JM, Schneeweiss S, Polinski JM, Rassen JA. Plasmode simulation for the evaluation of pharmacoepidemiologic methods in complex healthcare databases. *Comput Stat Data Anal* 2014;72:219–226. doi:10.1016/j.csda.2013.10.018
- Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, Robins J. Double/debiased machine learning for treatment and structural parameters. *Econom J* 2018;21:C1–C68. doi:10.1111/ectj.12097
- van der Laan MJ, Polley EC, Hubbard AE. Super learner. *Stat Appl Genet Mol Biol* 2007;6:Article 25. doi:10.2202/1544-6115.1309
- Zhu Z et al. Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets. *Nat Genet* 2016;48:481–487. doi:10.1038/ng.3538 (SMR/HEIDI)
