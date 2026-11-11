"""Pre-specified decision rules (paper 1: spec v1 section 5, spec v2 section 3, amendments A5 section 5 and A6 section 3).

All rules act on odds-ratio-scale bounds unless noted. The margin is a doubling (OR 2; Minikel et al. 2024's
benchmark for genetic support of drug targets).
"""
from __future__ import annotations

import numpy as np

__all__ = ["network_rule", "four_way_rule", "gate_rule", "validity_rule", "band_rule", "robustness_rule",
           "coprimary_reading", "MARGIN", "NC_BAND", "VALIDITY_THRESHOLD", "ROBUSTNESS_DLN", "GATE_MIN_POWER"]

MARGIN = 2.0
NC_BAND = (0.035, 0.065)
VALIDITY_THRESHOLD = 0.065
ROBUSTNESS_DLN = 0.2


def network_rule(upper, margin=MARGIN):
    """Network contrasts (E1, E2): "doubling excluded" if the profile 95% upper bound is below the margin,
    otherwise "inconclusive" (also when the bound is missing). Never "no enrichment"."""
    return "doubling excluded" if upper < margin else "inconclusive"


def four_way_rule(lo, hi, margin=MARGIN):
    """Co-primary contrasts (E5a, E5c), on the Wald 97.5% interval:
    lower > 1 and upper >= margin -> "enriched"; lower > 1 and upper < margin -> "modest enrichment";
    lower <= 1 and upper < margin -> "no material enrichment"; lower <= 1 and upper >= margin -> "inconclusive";
    a missing bound -> "failed"."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "failed"
    if lo > 1:
        return "enriched" if hi >= margin else "modest enrichment"
    return "no material enrichment" if hi < margin else "inconclusive"


GATE_MIN_POWER = 0.80


def gate_rule(lo, power, calibrated=True, min_power=GATE_MIN_POWER):
    """Positive-control gate (E5g; A5 section 5 with A6 section 3).

    The gate is evaluated only if the positive control is calibrated (validity rule) and G1's plasmode power at
    OR 2.7 (rejection rate at P < 0.05) is at least ``min_power``; otherwise it is "uninformative". When
    evaluated, it passes if G1's Wald 95% lower bound ``lo`` (OR scale) exceeds 1 and fails otherwise; a
    missing bound fails.
    """
    if not calibrated or not (np.isfinite(power) and power >= min_power):
        return "uninformative"
    return "pass" if (np.isfinite(lo) and lo > 1) else "fail"


def validity_rule(type1_n0, type1_n1, threshold=VALIDITY_THRESHOLD):
    """An estimator is calibrated for a design if its plasmode type I error at P < 0.05 is at most ``threshold``
    under both N0 and N1."""
    return bool(type1_n0 <= threshold and type1_n1 <= threshold)


def band_rule(rejection, band=NC_BAND):
    """A matched negative control passes if its rejection rate lies in ``band``."""
    return bool(band[0] <= rejection <= band[1])


def robustness_rule(d_ln_or, threshold=ROBUSTNESS_DLN):
    """A robustness tier passes if it moves the log odds ratio by less than ``threshold``."""
    return bool(abs(d_ln_or) < threshold)


def coprimary_reading(lo975, hi975, calibrated, nc_passes, gate=None, margin=MARGIN):
    """Final reading of a co-primary (A5 section 5 with A6 section 3): not interpreted unless calibrated and its
    negative control passes; otherwise the four-way rule, with "no material enrichment" read as
    "not interpretable as absence" if the positive-control gate failed.

    ``gate`` is the result of :func:`gate_rule` ("pass", "fail" or "uninformative", any case) or None if no
    gate was run. As registered, only a failed gate downgrades the reading; an uninformative gate leaves it
    unchanged, so report the gate's status beside the reading.
    """
    g = None if gate is None else str(gate).strip().lower()
    if g not in (None, "pass", "fail", "uninformative"):
        raise ValueError(f"gate must be 'pass', 'fail', 'uninformative' or None, not {gate!r}")
    if not (calibrated and nc_passes):
        return "not interpreted"
    r = four_way_rule(lo975, hi975, margin)
    if r == "no material enrichment" and g == "fail":
        return "not interpretable as absence (gate failed)"
    return r
