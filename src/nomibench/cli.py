"""Command line: ``nomibench {evaluate,calibrate,controls,d1} --data genes.csv ...``.

The gene table is a CSV (or .tsv/.parquet) with one row per testable gene. Results print as a table; ``--out``
writes them to CSV.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import __version__
from .aipw import d1
from .benchmark import calibrate, evaluate, matched_controls


def _read(path):
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError as err:
            raise ValueError("reading .parquet needs pyarrow: pip install 'nomibench[parquet]'") from err
    return pd.read_csv(path, sep="\t" if path.endswith((".tsv", ".tsv.gz")) else ",")


def _common(p):
    p.add_argument("--data", required=True, help="gene table (.csv, .tsv or .parquet)")
    p.add_argument("--outcome", required=True, help="0/1 column, e.g. SMR/HEIDI support")
    p.add_argument("--exposure", required=True, help="0/1 column: the nomination")
    p.add_argument("--covariates", required=True, help="comma-separated discoverability covariates")
    p.add_argument("--linear", default="", help="covariates entered linearly in the propensity model")
    p.add_argument("--out", default=None, help="write results to this CSV")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nomibench", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=f"nomibench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("evaluate", help="G1 and secondary estimators")
    _common(pe)
    pe.add_argument("--locus", default=None, help="column of GWAS-locus interval ids (empty outside loci)")
    pe.add_argument("--locus-distance", default=None, help="column of log10(distance to index variant + 1 kb)")
    pc = sub.add_parser("calibrate", help="plasmode calibration (the exposure's outcomes are not read)")
    _common(pc)
    pc.add_argument("--generator-covariates", required=True)
    pc.add_argument("--locus", default=None)
    pc.add_argument("--locus-distance", default=None)
    pc.add_argument("--targeting", default="estimand", choices=["estimand", "none"])
    pc.add_argument("--n-null", type=int, default=2000)
    pc.add_argument("--n-power", type=int, default=1000)
    pc.add_argument("--seed", type=int, required=True)
    pc.add_argument("--workers", type=int, default=1)
    pn = sub.add_parser("controls", help="discoverability-matched negative-control sets")
    _common(pn)
    pn.add_argument("--n-sets", type=int, default=1000)
    pn.add_argument("--seed", type=int, required=True)
    pn.add_argument("--workers", type=int, default=1)
    pd1 = sub.add_parser("d1", help="cross-fitted AIPW ATT risk ratio (median over splits)")
    _common(pd1)
    pd1.add_argument("--splits", type=int, default=50)
    pd1.add_argument("--seed", type=int, required=True)
    pd1.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    try:
        return _run(a)
    except (ValueError, FileNotFoundError) as err:      # bad input: a clear message, not a traceback
        print(f"nomibench {a.cmd}: error: {err}", file=sys.stderr)
        return 2


def _run(a):
    df = _read(a.data)
    cov = [c for c in a.covariates.split(",") if c]
    lin = [c for c in a.linear.split(",") if c]
    if a.cmd == "evaluate":
        res = evaluate(df, a.outcome, a.exposure, cov, lin, locus=a.locus, locus_distance=a.locus_distance)
        print(res.to_string(index=False))
        print(json.dumps(res.attrs, indent=1))
    elif a.cmd == "calibrate":
        gen = [c for c in a.generator_covariates.split(",") if c]
        cal = calibrate(df, a.outcome, a.exposure, cov, gen, lin, locus=a.locus, locus_distance=a.locus_distance,
                        targeting=a.targeting, n_null=a.n_null, n_power=a.n_power, seed=a.seed, workers=a.workers)
        res = cal.summary
        print(res.to_string(index=False))
        print(f"G1 calibrated (type I error <= 0.065 under N0 and N1): {cal.calibrated}")
    elif a.cmd == "controls":
        res, s = matched_controls(df, a.outcome, a.exposure, cov, lin, n_sets=a.n_sets, seed=a.seed, workers=a.workers)
        print(json.dumps(s.__dict__, indent=1))
    else:
        r = d1(df, a.outcome, a.exposure, cov, lin, splits=a.splits, entropy=a.seed, workers=a.workers)
        res = pd.DataFrame([r.__dict__])
        print(res.to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
