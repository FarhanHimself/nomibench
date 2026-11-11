"""Golden-output regression: the main entry points on a fixed synthetic table reproduce stored values."""
import json
import os

import numpy as np

from conftest import REF


def _cmp(a, b, rtol=1e-7):
    a = np.array([[np.nan if v is None else v for v in r] for r in a], float)
    b = np.array([[np.nan if v is None else v for v in r] for r in b], float)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-10, equal_nan=True)


def test_golden_outputs():
    from reference.make_golden import golden
    want = json.load(open(os.path.join(REF, "golden.json")))
    got = golden()
    assert got["evaluate_labels"] == want["evaluate_labels"]
    assert got["calibrate_cols"] == want["calibrate_cols"]
    for k in ("evaluate", "calibrate", "controls"):
        _cmp(got[k], want[k])
    # A block-bootstrap resample repeats genes, so its logit propensity scores contain exact ties. The spline
    # knots of G1's design are placed on the unique values, and BLAS rounding can split a tie by one ulp on
    # another platform, moving a knot. Replicates then agree only to about 1e-4 across environments; within an
    # environment they are exactly reproducible (test_controls_aipw_bootstrap.py).
    _cmp(got["bootstrap"], want["bootstrap"], rtol=1e-3)
    np.testing.assert_allclose(got["generator"], want["generator"], rtol=1e-9)
    np.testing.assert_allclose(got["betas"], want["betas"], rtol=1e-9, atol=1e-12)
    # D1's gradient-boosted trees are deterministic given the seed, but their splits change between
    # scikit-learn releases (1.8.0 -> 1.9.1 moved this RR by 0.4% and its P from 0.063 to 0.060). Exact
    # agreement is required only under the scikit-learn minor version that wrote the golden file.
    same = got["sklearn_version"].split(".")[:2] == want["sklearn_version"].split(".")[:2]
    if same:
        np.testing.assert_allclose(got["d1"], want["d1"], rtol=1e-6)
    else:
        np.testing.assert_allclose(got["d1"][:3], want["d1"][:3], rtol=0.05)
