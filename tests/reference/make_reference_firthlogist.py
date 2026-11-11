"""Reference values from firthlogist 0.5.0 (J. Luo; MIT licence), an independent implementation that follows
R logistf (its own tests match logistf's sex2 example at rtol 1e-5).

firthlogist needs scikit-learn < 1.6; run in a separate environment:
  python -m venv ref && ref/bin/pip install "scikit-learn==1.5.2" "numpy<2.1" scipy pandas tabulate
  ref/bin/pip install --no-deps --ignore-requires-python firthlogist==0.5.0
  ref/bin/python tests/reference/make_reference_firthlogist.py
Writes tests/reference/firthlogist.json. Coefficients are ordered (intercept, covariates...), the order nomibench
uses when the design's first column is the intercept.
"""
import json
import os

import numpy as np
import pandas as pd
import firthlogist
from firthlogist import FirthLogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SETS = {"moderate": ["x1", "x2", "x3"], "quasi_separation": ["b", "c"], "rare_events": ["e", "z", "z2"],
        "separation": ["u", "v"]}
out = {"_source": f"firthlogist {firthlogist.__version__ if hasattr(firthlogist, '__version__') else '0.5.0'}; "
                  "tol 1e-10, max_iter 1000, pl_max_iter 1000; profile penalised-likelihood 95% CI; penalised LRT P"}
for name, cols in SETS.items():
    d = pd.read_csv(os.path.join(DATA, f"firth_{name}.csv"))
    fl = FirthLogisticRegression(max_iter=1000, pl_max_iter=1000, tol=1e-10, max_stepsize=5, pl_max_stepsize=5)
    fl.fit(d[cols].values.astype(float), d.y.values.astype(float))
    order = [len(cols)] + list(range(len(cols)))                  # intercept first
    coef = np.r_[fl.intercept_, fl.coef_]
    out[name] = dict(columns=["(Intercept)"] + cols, coef=coef.tolist(), loglik=float(fl.loglik_),
                     ci=np.asarray(fl.ci_)[order].tolist(), pvalue=np.asarray(fl.pvals_)[order].tolist(),
                     n_iter=int(fl.n_iter_) if hasattr(fl, "n_iter_") else None)
# G1's actual design [1, E, s(logit PS)] (tests/data/g1_design.csv), fitted as given: no added intercept
g = pd.read_csv(os.path.join(DATA, "g1_design.csv"))
cols = ["c0", "c1", "c2", "c3", "c4"]
fl = FirthLogisticRegression(fit_intercept=False, max_iter=1000, pl_max_iter=1000, tol=1e-10, max_stepsize=5,
                             pl_max_stepsize=5)
fl.fit(g[cols].values.astype(float), g.y.values.astype(float))
out["g1_design"] = dict(columns=cols, coef=np.asarray(fl.coef_).tolist(), loglik=float(fl.loglik_),
                        ci=np.asarray(fl.ci_).tolist())
with open(os.path.join(HERE, "firthlogist.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: (v["coef"] if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
