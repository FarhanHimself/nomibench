"""Golden outputs of the main entry points on a fixed synthetic gene table (tests/test_regression.py).

Regenerate only for an intended numerical change, and record why in CHANGELOG.md:
    python tests/reference/make_golden.py
"""
import json
import os
import sys

import numpy as np
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from conftest import make_genes                                  # noqa: E402
import nomibench as nb                                           # noqa: E402
from nomibench import aipw, bootstrap, negative_control           # noqa: E402

COV = ["x_z", "x_len", "x_nd"]
ALL_ESTIMATORS = ("g1", "naive", "g1b", "g1c", "h", "w1", "equiv", "g1_profile")


def _num(df):
    """Numeric columns as nested lists, NaN as None (JSON)."""
    return [[None if (isinstance(v, float) and np.isnan(v)) else v for v in row]
            for row in df.select_dtypes("number").values.tolist()]


def golden():
    d = make_genes(n=3000, seed=101, effect=0.5)
    ev = nb.evaluate(d, "Y", "E", COV, linear=("x_nd",), locus="interval", locus_distance="ld")
    cal = nb.calibrate(d, "Y", "E", COV, ["x_z", "x_len"], linear=("x_nd",), locus="interval", locus_distance="ld",
                       n_null=4, n_power=2, power_ors=(2.0,), seed=5, estimators=ALL_ESTIMATORS)
    reps = cal.replicates.drop(columns=["rep"])
    nc = negative_control.run(d, "Y", "E", COV, ("x_nd",), n_sets=4, seed=6)
    bs = bootstrap.run(d, "Y", ["E"], COV, ("x_nd",), chrom="chr", tss="tss", n_reps=3, seed=7)
    r = aipw.d1(d, "Y", "E", COV, ("x_nd",), splits=2, entropy=8)
    return dict(evaluate=_num(ev[["OR", "lo", "hi", "P"]]), evaluate_labels=ev.estimator.tolist(),
                calibrate=_num(reps), calibrate_cols=list(reps.select_dtypes("number").columns),
                generator=cal.generator.tolist(), betas=[s.beta for s in cal.scenarios],
                controls=_num(nc), bootstrap=_num(bs), d1=[r.rr, r.lo, r.hi, r.pvalue],
                sklearn_version=sklearn.__version__)


if __name__ == "__main__":
    g = golden()
    with open(os.path.join(HERE, "golden.json"), "w") as f:
        json.dump(g, f, indent=1)
    print("written", {k: len(v) for k, v in g.items()})
