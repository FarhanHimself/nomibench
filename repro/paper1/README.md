# Reproducing paper 1

Paper 1 benchmarks Alzheimer's disease gene nominations against SMR/HEIDI support. The support comes from
MetaBrain cortex-EUR cis-eQTL and the 2026 consensus AD GWAS. Its gene-level inputs are derived from MetaBrain,
and MetaBrain's terms do not allow results to be redistributed. **No data ship here.** You rebuild the inputs
from your own copies, then check them against the registered numbers.

| File | What it does |
|---|---|
| `build_covariates.py` | Builds the U_gw covariate table: discoverability covariates (Tier P), GWAS-locus assignment and exposures E1–E3. It is a port of the registered `19a_covariates.py` with path arguments; its logic is unchanged |
| `build_e5_sets.py` | Builds the external nomination sets E5 from Agora and Open Targets. It is the registered `20a_e5_exposures.py`, unchanged |
| `reproduce_registered.py` | Re-runs the registered step 19–20 analyses with `nomibench` and compares every number with the frozen outputs (criteria R1–R9, plus R10 added after the specification). It writes aggregates only |

## Inputs and their terms

| Input | Source | Terms |
|---|---|---|
| MetaBrain cortex-EUR cis-eQTL, release 2021-07-23 (80 PCs): per-chromosome files and TopEffects | metabrain.nl | by request; results may not be redistributed |
| AD GWAS, main release | GWAS Catalog GCST90704646 (Nat Genet 2026; doi:10.1038/s41588-026-02583-1) | open |
| GWAS loci (Table S3 of the same paper) | journal supplement | open |
| LD reference | 1000 Genomes phase 3, EUR, per chromosome | open |
| Gene annotation | GENCODE v42 basic | open |
| RADC local regulatory networks (`LRN_*`) | RADC Resource Sharing Hub (Tasaki et al. 2018, Front Genet 9:294) | RADC terms |
| Agora nominated targets | Synapse syn13363290, version 119 | open, registration |
| Open Targets AD associations by data type | Open Targets Platform 26.09 | open |
| LOEUF; brain BA9 expression and tissue specificity (tier S2 only) | gnomAD v4.1.1; GTEx v11 | open |

## Steps

1. **eQTL BESD.** Convert the MetaBrain cis-eQTL statistics to SMR's BESD format, keeping associations with
   P < 1.57×10⁻³ (GRCh38).
2. **SMR/HEIDI.** For each chromosome, run SMR 1.3.1 with default HEIDI:

   ```
   smr --bfile 1kg_eur_chrN --gwas-summary ad_gwas.ma --beqtl-summary metabrain \
       --chr N --peqtl-smr 5e-8 --cis-wind 2000 --out smr_chrN
   ```

   The 1000 Genomes `.bim` positions were relabelled to GRCh38 through rsIDs; the genotypes were not lifted
   over. Concatenate the chromosomes into `smr_gw_all.csv`.
3. **Outcome.** This is built inside `reproduce_registered.py`. A gene is supported if BH q(p_SMR) < 0.05 over all
   11,224 probes and p_HEIDI > 0.05; a missing HEIDI counts as unsupported.
4. **Covariates.** `python build_covariates.py --smr smr_gw_all.csv --topeffects ... --gencode ... --table-s3 ...
   --lrn ... --loeuf ... --gtex ... --out ugw_covariates.csv`
5. **External sets.** `python build_e5_sets.py --agora ... --ot ... --out .`
6. **Check.** Run `python reproduce_registered.py --cov ugw_covariates.csv --smr smr_gw_all.csv --sets
   e5_exposure_sets_A5.tsv --step19 DIR --step20 DIR --step25 DIR --out OUT --require-frozen
   --parts R1,R2,R3,R4,R4b,R5,R6,R7,R8,R9,R10`. The `--step19`, `--step20` and `--step25` folders hold the frozen
   aggregate outputs of paper 1 (simulation replicates, control sets and result tables; no per-gene values).
   `--step25` needs only `a10_e2_plasmode_design.csv`, `a10_e2_plasmode_raw_*.csv`, `a10_nc_e2.csv` and
   `a10_boot_5Mb.csv`; the step's `private/` folder holds a per-gene table and is never released.

With `--require-frozen` the script stops unless the three inputs match the frozen SHA-256 values in
`nomibench.registered.FROZEN_SHA256`; without it, a mismatch is reported and the comparisons show where the
outputs differ. Every comparison is NaN-aware: a value missing on one side only counts as an infinite difference.

## What was checked, and the result

The criteria were fixed before the run (step 29 spec, 6 Oct 2026). The results are in `docs/VALIDATION.md`.

| | Checked | Tolerance |
|---|---|---|
| R1 | the ten registered G1 estimates | \|Δ ln OR\| ≤ 1e-8 |
| R2 | every non-D1 row of the step 19 and step 20 result tables (83 rows) | estimate 1e-6 relative; bounds 1e-5 on the log scale; P 1e-5 relative |
| R3 | estimand-targeting constants and generator coefficients; the estimand achieved by the package's β\*, recomputed | 1e-8 |
| R4 | the first 50 replicates of every frozen plasmode file (46 files), regenerated from the registered seeds | events identical; 1e-8 |
| R5 | plasmode summaries recomputed from the frozen replicate files | 1e-12 |
| R6 | the first 100 matched control sets of E4, NC_E5a and NC_E5c | events identical; 1e-8 |
| R7 | D1 on the real outcome (50 splits) | 1e-6 on the log scale |
| R8 | the decision rules applied to the registered estimates | identical readings |
| R9 | the convergence flag at the registered and the default iteration cap | as expected; Δ ln OR ≤ 1e-7 |
| R10 (post-spec) | amendment A10: β\* of the E2-design plasmode; its first 50 replicates per scenario (G1, profile bounds); the first 100 E2-matched control sets; the first 50 replicates of the 5-Mb block bootstrap | 1e-8; bootstrap 1e-3 (tied propensity scores in resamples can move a spline knot across platforms) |

R4b (D1 on plasmode replicates) and R7b (D1 for E1) are informational: their frozen values came from a laptop run.

**Environment.** Step 20 ran under Python 3.11.15 with numpy 2.4.4, scipy 1.17.1, pandas 3.0.2,
statsmodels 0.15.0, patsy 1.0.3 and scikit-learn 1.8.0. In that environment the replicate-level checks agree to
about 1e-15. Under pandas 2.3 one matched-control set differs by 1.4e-8 in ln OR. Its Firth fit stops on a
1e-9 step tolerance along a nearly flat direction, so values at that scale depend on upstream rounding. No
estimate, interval or decision changes.
