"""Shared fixtures: a synthetic gene table shaped like a nomination benchmark (no real genes)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REF = os.path.join(HERE, "reference")


def make_genes(n=4000, seed=7, effect=0.0, n_loci=40, locus_share=0.06):
    """Synthetic testable genes.

    Discoverability covariates x_z (eQTL strength), x_len (length), x_nd (a small count, entered linearly) drive
    both the nomination E and the outcome Y; a share of genes sits in GWAS-locus intervals, where Y is more
    frequent and E is mildly enriched. ``effect`` is the true log OR of E on Y given everything else.
    """
    rng = np.random.default_rng(seed)
    x_z = rng.normal(size=n)
    x_len = rng.normal(size=n)
    x_nd = rng.integers(1, 6, n).astype(float)
    in_locus = rng.random(n) < locus_share
    interval = np.where(in_locus, rng.integers(0, n_loci, n), -1)
    dist = rng.uniform(1e3, 5e5, n)
    ld = np.where(in_locus, np.log10(dist + 1e3), 0.0)
    pe = 1 / (1 + np.exp(-(-2.2 + 0.7 * x_z + 0.3 * x_len + 0.1 * x_nd + 0.6 * in_locus)))
    E = rng.binomial(1, pe).astype(float)
    eta = -4.2 + 0.8 * x_z + 0.2 * x_len + 2.3 * in_locus - 0.3 * in_locus * (ld - 5) + effect * E
    Y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    return pd.DataFrame({"gene": [f"G{i:05d}" for i in range(n)], "x_z": x_z, "x_len": x_len, "x_nd": x_nd,
                         "L": in_locus, "interval": pd.Series(np.where(in_locus, interval, np.nan)),
                         "ld": ld, "E": E, "Y": Y, "chr": rng.integers(1, 23, n), "tss": rng.uniform(1, 2.4e8, n)})


@pytest.fixture(scope="session")
def genes():
    return make_genes()


@pytest.fixture(scope="session")
def firth_ref():
    with open(os.path.join(REF, "firthlogist.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def r_ref():
    with open(os.path.join(REF, "r_reference.json")) as f:
        return json.load(f)


def load_firth_set(name):
    d = pd.read_csv(os.path.join(DATA, f"firth_{name}.csv"))
    cols = [c for c in d.columns if c != "y"]
    X = np.c_[np.ones(len(d)), d[cols].values.astype(float)]
    return X, d.y.values.astype(float)
