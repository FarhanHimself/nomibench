# Test data

All files are small fixtures for the test suite. None contains real genes or MetaBrain-derived values.

| File | Source |
|---|---|
| `firth_*.csv`, `conditional.csv`, `mh_tables.csv` | Simulated by `tests/reference/make_synthetic_data.py` (fixed seeds) |
| `g1_design.csv` | Simulated genes; G1's design matrix built by nomibench (statsmodels propensity score, patsy spline), exported so that R brglm2 and firthlogist fit the identical matrix |
| `logistf_sex2.csv` | The `sex2` example data of the R package logistf (Heinze, Ploner, Jiricka and Steiner), as redistributed by firthlogist 0.5.0 (MIT licence) |
| `logistf_endometrial.csv` | The `endometrial` example data (Heinze and Schemper 2002, *Stat Med* 21:2409) as distributed with logistf and redistributed by firthlogist 0.5.0 |

The two logistf datasets are included only so that the tests can compare nomibench's Firth estimates with the published logistf results for them.
