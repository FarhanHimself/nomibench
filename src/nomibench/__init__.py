"""nomibench: calibrated, discoverability-matched benchmarks of gene nominations against genetic support.

Does a set of nominated genes (from a network, a curated resource, an omics screen) carry more genetic causal
support, such as SMR/HEIDI or colocalisation support, than its genes' detectability predicts? Genes that are easy
to instrument (strong, well-powered eQTLs; long, gene-dense, well-studied) are both more often nominated and more
often "supported", so an unmatched comparison rewards discoverability. The package provides:

* G1, the primary estimator: Firth logistic regression of support on the nomination and a spline of the logit
  propensity score of nomination given discoverability, with penalised-likelihood tests and intervals;
* secondary estimators (propensity-quintile Mantel-Haenszel, observed/expected, within-locus conditional Firth,
  a doubly robust cross-fitted AIPW risk ratio);
* plasmode calibration with estimand-targeted null and power scenarios;
* discoverability-matched negative-control gene sets on the real outcome;
* a genomic-block bootstrap and the pre-specified decision rules.

Modules: :mod:`~nomibench.firth`, :mod:`~nomibench.conditional`, :mod:`~nomibench.tables`,
:mod:`~nomibench.propensity`, :mod:`~nomibench.estimators`, :mod:`~nomibench.plasmode`,
:mod:`~nomibench.negative_control`, :mod:`~nomibench.aipw`, :mod:`~nomibench.bootstrap`,
:mod:`~nomibench.decisions`, :mod:`~nomibench.registered`; one-call entry points in :mod:`~nomibench.benchmark`.
"""
from importlib.metadata import PackageNotFoundError, version as _version

from . import aipw, bootstrap, conditional, decisions, estimators, firth, negative_control, plasmode, propensity, \
    registered, tables
from ._util import ConvergenceWarning
from .benchmark import Calibration, calibrate, evaluate, matched_controls
from .estimators import g1, g1_design
from .firth import firth_fit, firth_lrt, firth_profile_ci
from .propensity import fit_ps

try:
    __version__ = _version("nomibench")
except PackageNotFoundError:          # running from a source tree
    __version__ = "1.0.0"

__all__ = ["__version__", "ConvergenceWarning", "evaluate", "calibrate", "matched_controls", "Calibration",
           "g1", "g1_design", "fit_ps", "firth_fit", "firth_lrt", "firth_profile_ci", "aipw", "bootstrap",
           "conditional", "decisions", "estimators", "firth", "negative_control", "plasmode", "propensity",
           "registered", "tables"]
