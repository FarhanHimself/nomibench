#!/usr/bin/env python3
"""
reproduce_registered.py -- re-run paper 1's registered step 19-20 analyses with nomibench and compare every
number with the frozen outputs (step 29 spec, section 3: criteria R1-R9).

PRIVATE INPUTS (MetaBrain-derived; never distributed): the U_gw covariate table and the SMR table. The script
writes aggregates only: differences, counts and pass/fail. No per-gene value is written or printed.

  python reproduce_registered.py --cov ugw_covariates.csv --smr smr_gw_all.csv --sets e5_exposure_sets_A5.tsv \
      --step19 DIR --step20 DIR --out OUTDIR [--step25 DIR] [--parts R1,...,R10] [--workers 2] [--require-frozen]

R1-R9 are the step-29 spec's criteria. R4b (D1 on plasmode replicates) and R7b (E1 D1) are informational: the frozen
values came from a laptop run. R10 was added after the spec and is labelled post-spec: it regenerates the
amendment A10 outputs (E2-design plasmode, E2-matched negative control, 5-Mb genomic-block bootstrap) and needs
--step25. Every comparison is NaN-aware: a value missing on one side only counts as an infinite difference.
--require-frozen stops if an input's sha256 differs from the frozen one; without it the mismatch is reported and
the comparisons will show where the outputs differ.

--step19 holds gw19_results.csv, gw19_plasmode.csv, gw19_plasmode_raw_*.csv, gw19_e4_sets.csv, d1_results.csv and
d1_plasmode_raw_N0.csv. --step20 holds real/e5_results.csv, real/e5_decisions.csv and cal/ (e5cal_design*.csv,
e5cal_plasmode*.csv, e5cal_raw_*.csv, e5cal_asw_raw_*.csv, e5cal_nc_*.csv). --step25 holds a10_e2_plasmode_design.csv,
a10_e2_plasmode_raw_*.csv, a10_nc_e2.csv and a10_boot_5Mb.csv.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import nomibench as nb
from nomibench import aipw, decisions, negative_control, plasmode as pm, registered as REGD
from nomibench.estimators import (g1, g1_design, g1b, g1c, h_design, h_interaction, naive, unadjusted_firth, w1, w2)
from nomibench.propensity import balance, fit_ps

warnings.filterwarnings("ignore", category=nb.ConvergenceWarning)   # registered maxit=100 binds for E1 (R9)

TIERP, LIN, MAXIT = list(REGD.TIER_P), REGD.TIER_P_LINEAR, REGD.MAXIT
REG = {"E1": 1.3421317211467318, "E1w": 1.358688072518047, "E2": 1.0279495452090348, "E5a": 0.9819827047006083,
       "E5c": 1.6404346532697651, "E5g": 3.4671675118848864, "E5b": 3.6343775335011737, "E5e": 1.2713501903708453,
       "E5d": 1.4714718337130983, "E5c_naive": 1.8641610729269793}       # A10 part 0, full precision
ORDER = ["E1", "E1w", "E2", "E5a", "E5c", "E5g", "E5b", "E5e", "E5d", "E5c_naive"]
BOOL = ["L", "E1", "E2", "E3", "tierP_complete", "mhc_apoe"]

def main():
    ap = argparse.ArgumentParser()
    for k in ("cov", "smr", "sets", "step19", "step20", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--step25", default=None, help="A10 output directory (needed for R10 only)")
    ap.add_argument("--parts", default="R1,R2,R3,R4,R4b,R5,R6,R7,R8,R9")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reps", type=int, default=50, help="replicates per raw file (R4, R10)")
    ap.add_argument("--sets-n", type=int, default=100, help="matched sets per control (R6, R10)")
    ap.add_argument("--require-frozen", action="store_true", help="stop if an input differs from the frozen file")
    A = ap.parse_args()
    if "R10" in A.parts.split(",") and not A.step25:
        ap.error("R10 needs --step25")
    PARTS = set(A.parts.split(","))
    os.makedirs(A.out, exist_ok=True)
    LOG = open(os.path.join(A.out, "repro_report.txt"), "a", encoding="utf-8")
    CHECKS = []


    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        LOG.write(s + "\n")
        LOG.flush()


    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()


    def check(part, name, diff, tol, info=False, n=None):
        ok = bool(np.isfinite(diff) and diff <= tol) if not info else None
        CHECKS.append(dict(part=part, check=name, max_abs_diff=float(diff), tol=tol, n=n,
                           result=("info" if info else ("PASS" if ok else "FAIL"))))
        say(f"  [{'info' if info else ('PASS' if ok else 'FAIL')}] {part} {name}: max |diff| {diff:.3g} (tol {tol:g})"
            + (f" over {n}" if n is not None else ""))
        return ok


    def adiff(a, b):
        """|a - b|, NaN-aware: 0 if both are missing, infinite if exactly one is (never silently dropped)."""
        a, b = float(a), float(b)
        if np.isnan(a) or np.isnan(b):
            return 0.0 if (np.isnan(a) and np.isnan(b)) else np.inf
        return abs(a - b)


    def vdiff(a, b):
        """Largest NaN-aware |a - b| over two aligned arrays (0 for empty)."""
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        both, one = np.isnan(a) & np.isnan(b), np.isnan(a) ^ np.isnan(b)
        if one.any():
            return np.inf
        d = np.abs(a - b)[~both]
        return float(d.max()) if d.size else 0.0


    def rel(a, b):
        d = adiff(a, b)
        return d if not np.isfinite(d) or d == 0 else d / max(abs(float(b)), 1e-300)


    def dlog(a, b):
        a, b = float(a), float(b)
        return adiff(np.log(a) if a > 0 else np.nan, np.log(b) if b > 0 else np.nan)


    # ------------------------------------------------------------------ inputs (gate)
    t0 = time.time()
    say("=" * 78)
    say(f"nomibench {nb.__version__} | reproduction of paper 1 (step 29) | {time.strftime('%Y-%m-%d %H:%M')}")
    say(f"python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | "
        + " | ".join(f"{m} {__import__(m).__version__}" for m in ("scipy", "statsmodels", "patsy", "sklearn")))
    say("=" * 78)
    frozen = True
    for p, k in ((A.cov, "ugw_covariates.csv"), (A.smr, "smr_gw_all.csv"), (A.sets, "e5_exposure_sets_A5.tsv")):
        h = sha(p)
        ok = h == REGD.FROZEN_SHA256[k]
        frozen &= ok
        say(f"input {k}: sha256 {h[:12]}... " + ("matches the freeze" if ok else "DOES NOT MATCH the freeze"))
        if not ok and A.require_frozen:
            sys.exit(f"{k}: sha256 {h} does not match the frozen input (--require-frozen)")
    if not frozen:
        say("WARNING: an input differs from the frozen file, so the comparisons below are not a reproduction")

    X_all = pd.read_csv(A.cov, dtype={"ensg": str, "chr": str, "chr_use": str, "ProbeChr": str}, float_precision="round_trip")
    for c in BOOL:
        X_all[c] = X_all[c].astype(str).str.lower().isin(["true", "1"])
    D = X_all[X_all.tierP_complete].reset_index(drop=True)


    def load_y(kind):
        d = pd.read_csv(A.smr, usecols=["probeID", "p_SMR", "p_HEIDI"])
        p = d.p_SMR.values.astype(float); n = len(p); o = np.argsort(p)
        q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        h = d.p_HEIDI.values.astype(float)
        y = ((q < 0.05) & (h > 0.05)) if kind == "fdr" else ((p < 0.05 / n) & ((h > 0.05) | np.isnan(h)))
        return pd.Series(y.astype(float), index=d.probeID.values)


    YALL, YBON = load_y("fdr"), load_y("bonf")
    S = pd.read_csv(A.sets, sep="\t", dtype={"ensembl_id": str}, keep_default_na=False)
    say(f"analysis set {len(D):,} genes; in-locus {int(D.L.sum())}; outcome {int(YALL.sum())} supported")


    def e5_set(ex):
        """Paper 1 A5 analysis set: U_gw with complete Tier P (E5c_naive: genes not in E5g, both arms)."""
        ids = lambda col: set(S.loc[S[col] == 1, "ensembl_id"])
        Ds = D[~D.ensg.isin(ids("E5g"))] if ex == "E5c_naive" else D
        Ds = Ds.reset_index(drop=True).copy()
        Ds["E"] = Ds.ensg.isin(ids(ex)).astype(float)
        return Ds


    def design_set(ex):
        if ex == "E1":
            Ds = D.copy(); Ds["E"] = Ds.E1.astype(float)
        elif ex == "E1w":
            Ds = D[D.E2.astype(bool)].reset_index(drop=True).copy(); Ds["E"] = Ds.E1.astype(float)
        elif ex == "E2":
            Ds = D.copy(); Ds["E"] = Ds.E2.astype(float)
        else:
            Ds = e5_set(ex)
        Ds["Y"] = YALL.reindex(Ds.ensg).values
        assert not Ds.Y.isna().any()
        return Ds


    DS = {ex: design_set(ex) for ex in ORDER}
    PS = {ex: fit_ps(DS[ex], "E", TIERP, LIN) for ex in ORDER}
    G19 = pd.read_csv(os.path.join(A.step19, "gw19_results.csv")).set_index("analysis")
    E5R = pd.read_csv(os.path.join(A.step20, "real", "e5_results.csv"))
    E5D = pd.read_csv(os.path.join(A.step20, "real", "e5_decisions.csv")).set_index("exposure")
    CAL = os.path.join(A.step20, "cal")


    def cmp_row(part, label, got, want, est_tol=1e-6, b_tol=1e-5, p_tol=1e-5):
        """got/want = (OR, lo, hi, P); estimate relative, bounds on the log scale, P relative."""
        d = [rel(got[0], want[0])]
        for i in (1, 2):
            if not (np.isnan(want[i]) and np.isnan(got[i])):
                d.append(dlog(got[i], want[i]) * (est_tol / b_tol))   # scale bounds onto the estimate tolerance
        d.append(rel(got[3], want[3]) * (est_tol / p_tol))
        return max(d)


    # ------------------------------------------------------------------ R1
    if "R1" in PARTS:
        say("-" * 78); say("R1  the ten registered G1 estimates (maxit = 100)")
        mx = 0.0
        for ex in ORDER:
            r = g1(DS[ex].Y.values, DS[ex].E.values, PS[ex].logit, profile_levels=(), maxit=MAXIT)
            mx = max(mx, adiff(r.log_or, np.log(REG[ex])))
        check("R1", "ten G1 ln OR vs registered", mx, 1e-8, n=10)

    # ------------------------------------------------------------------ R2
    SRC = {"loeuf": "gnomad_loeuf", "ba9": "gtex_ba9_log2tpm", "tau": "gtex_tau"}


    def s2_ps(Ds, expo_col):
        """Registered S2 tier (v1 A5 rules): Tier P + log LOEUF + BA9 + tau; median imputation; indicators."""
        D2 = Ds.copy(); extra = []; cols2 = list(TIERP); Eb = D2[expo_col].values.astype(bool)
        for k, col in SRC.items():
            v = D2[col].astype(float); v = np.log(v.clip(lower=0.01)) if k == "loeuf" else v
            m = v.isna(); D2[f"x_{k}"] = v.fillna(v.median()); cols2.append(f"x_{k}")
            add = bool(m.sum() >= 10 and m.values[Eb].any() and m.values[~Eb].any()
                       and not any(D2[i].astype(bool).equals(m) for i in extra))
            if add:
                D2[f"m_{k}"] = m.astype(float); extra.append(f"m_{k}")
        return fit_ps(D2, expo_col, cols2, LIN, extra)


    if "R2" in PARTS:
        say("-" * 78); say("R2  every non-D1 row of gw19_results.csv and e5_results.csv")
        diffs = {}

        def rec(key, got, want, **k):
            diffs[key] = cmp_row("R2", key, got, want, **k)

        def g1row(y, e, lps, level=0.95):
            r = g1(y, e, lps, profile_levels=(level,), maxit=MAXIT)
            lo, hi = r.profile_or(level)
            return (r.odds_ratio, lo, hi, r.pvalue)

        W = lambda lab: tuple(G19.loc[lab, ["OR", "lo", "hi", "P"]].astype(float))
        Ds = DS["E1"]; y = Ds.Y.values; E1 = Ds.E.values; lps = PS["E1"].logit; inl = Ds.L.values
        iid = Ds.interval.fillna(-1).astype(int).values
        _, psq = balance(Ds, "E", TIERP, PS["E1"].ps)
        rec("E1 naive", tuple(naive(y, E1)), W("E1 naive Fisher"))
        rec("E1 unadjusted Firth", unadjusted_firth(y, E1, maxit=MAXIT), W("E1 unadjusted Firth"))
        rec("E1 G1", g1row(y, E1, lps), W("E1 G1  PS-adjusted Firth  <-- PRIMARY"))
        rec("E1 G1b", g1b(y, E1, np.asarray(psq)), W("E1 G1b PS-quintile Mantel-Haenszel"))
        XOE = np.c_[np.ones(len(Ds)), Ds[TIERP].values]
        oe, p2, _, _ = g1c(y, E1, XOE, maxit=MAXIT)
        rec("E1 G1c", (oe, np.nan, np.nan, p2), W("E1 G1c observed/expected (Poisson-binomial)"))
        rec("E1 H", h_interaction(y, E1, inl, lps, maxit=MAXIT), W("E1 H   E1 x L interaction (secondary)"))
        rec("E1 naive in-locus", tuple(naive(y[inl], E1[inl])), W("E1 naive Fisher, within-locus"))
        for lab, cov in (("E1 W1  CFL: E + log dist + logit PS", np.c_[Ds.ld.values[inl], lps[inl]]), ("E1 W1''CFL: E only", None)):
            wr = w1(y[inl], E1[inl], iid[inl], cov)
            rec(lab, (wr.odds_ratio, np.exp(wr.lo), np.exp(wr.hi), wr.pvalue), W(lab))
        rec("E1 W2", w2(y[inl], E1[inl], iid[inl]), W("E1 W2  Mantel-Haenszel by interval"))
        yb = YBON.reindex(Ds.ensg).values
        rec("E1 Bonferroni", g1row(yb, E1, lps), W("E1 G1 with Bonferroni outcome (sensitivity)"))
        rec("E1 S2", g1row(y, E1, s2_ps(Ds, "E").logit), W("E1 G1 tier S2 (P + LOEUF + BA9 + tau)"))
        keep = ~Ds.mhc_apoe.values; Dm = Ds[keep].reset_index(drop=True)
        rec("E1 MHC/APOE", g1row(y[keep], Dm.E.values, fit_ps(Dm, "E", TIERP, LIN).logit),
            W("E1 G1 excluding MHC and APOE regions (sensitivity)"))
        for ex, lab in (("E1w", "E1w"), ("E2", "E2")):
            Ds = DS[ex]; y = Ds.Y.values; e = Ds.E.values; lps = PS[ex].logit
            inl = Ds.L.values; iid = Ds.interval.fillna(-1).astype(int).values
            rec(f"{ex} naive", tuple(naive(y, e)), W(f"{lab} naive Fisher"))
            rec(f"{ex} G1", g1row(y, e, lps), W(f"{lab} G1  PS-adjusted Firth"))
            for suf, cov in (("W1  CFL: E + log dist + logit PS", np.c_[Ds.ld.values[inl], lps[inl]]), ("W1''CFL: E only", None)):
                wr = w1(y[inl], e[inl], iid[inl], cov)
                rec(f"{ex} {suf}", (wr.odds_ratio, np.exp(wr.lo), np.exp(wr.hi), wr.pvalue), W(f"{lab} {suf}"))
        e3 = X_all.E3.values; y3 = YALL.reindex(X_all.ensg).values
        k3, n3 = int(y3[e3].sum()), int(e3.sum()); bt = binomtest(k3, n3, float(y3[~e3].mean()))
        ci = bt.proportion_ci(0.95, method="exact")
        rec("E3", (k3 / n3, ci.low, ci.high, bt.pvalue), W("E3 GVC supported / testable"))
        R4f = pd.read_csv(os.path.join(A.step19, "gw19_e4_sets.csv"))
        s4 = negative_control.summarise(R4f)
        rec("E4 rate", (s4.reject, s4.reject - 1.96 * s4.mc_se, s4.reject + 1.96 * s4.mc_se, np.nan),
            W("E4 rejection rate at P<0.05"))
        # E5 sets
        for ex in REGD.E5_SETS:
            Ds = DS[ex]; y = Ds.Y.values; e = Ds.E.values; lps = PS[ex].logit
            inl = Ds.L.values; iid = Ds.interval.fillna(-1).astype(int).values
            Rx = E5R[E5R.exposure == ex].set_index("analysis")
            Wx = lambda lab: tuple(Rx.loc[lab, ["OR", "lo", "hi", "P"]].astype(float))
            rec(f"{ex} naive", tuple(naive(y, e)), Wx("naive Fisher"))
            r = g1(y, e, lps, profile_levels=(0.95, 0.975), wald_levels=(0.95, 0.975), maxit=MAXIT)
            for lev, lab in ((0.95, "G1 PS-Firth (profile 95% CI)"), (0.975, "G1 PS-Firth (profile 97.5% CI)")):
                rec(f"{ex} {lab}", (r.odds_ratio, *r.profile_or(lev), r.pvalue), Wx(lab))
            for lev, lab in ((0.95, "G1 PS-Firth (Wald 95% CI)"), (0.975, "G1 PS-Firth (Wald 97.5% CI)")):
                rec(f"{ex} {lab}", (r.odds_ratio, *r.wald_or(lev), r.pvalue), Wx(lab))
            _, psq = balance(Ds, "E", TIERP, PS[ex].ps)
            rec(f"{ex} G1b", g1b(y, e, np.asarray(psq)), Wx("G1b PS-quintile Mantel-Haenszel"))
            oe, p2, _, _ = g1c(y, e, np.c_[np.ones(len(Ds)), Ds[TIERP].values], maxit=MAXIT)
            rec(f"{ex} G1c", (oe, np.nan, np.nan, p2), Wx("G1c observed/expected (Poisson-binomial)"))
            wr = w1(y[inl], e[inl], iid[inl], np.c_[Ds.ld.values[inl], lps[inl]])
            rec(f"{ex} W1", (wr.odds_ratio, np.exp(wr.lo), np.exp(wr.hi), wr.pvalue), Wx("W1 within-locus conditional Firth"))
            if ex in REGD.E5_COPRIMARY:
                rec(f"{ex} S2", g1row(y, e, s2_ps(Ds, "E").logit), Wx("G1 tier S2 (P + LOEUF + BA9 + tau)"))
                keep = ~Ds.mhc_apoe.values; Dm = Ds[keep].reset_index(drop=True)
                rec(f"{ex} MHC/APOE", g1row(y[keep], Dm.E.values, fit_ps(Dm, "E", TIERP, LIN).logit),
                    Wx("G1 excluding MHC and APOE regions"))
        worst = max(diffs, key=diffs.get)
        check("R2", f"{len(diffs)} rows (estimate rel. 1e-6; bounds 1e-5 log; P rel. 1e-5; worst: {worst})",
              max(diffs.values()), 1e-6, n=len(diffs))
        pd.Series(diffs, name="scaled_max_diff").to_csv(os.path.join(A.out, "repro_R2_rows.csv"))

    # ------------------------------------------------------------------ shared: E5 calibration designs
    def e5_design(ex):
        Ds = DS[ex]; E = Ds.E.values; inl = Ds.L.values
        y = np.where(E == 0, Ds.Y.values, np.nan)                     # exposed outcomes never enter
        X_gen = np.c_[np.ones(len(Ds)), Ds.x_z, Ds.x_len, Ds.x_snp, Ds.L.astype(float), Ds.ld]
        gam = pm.fit_generator(X_gen, y, E, maxit=MAXIT)
        eta0, eta1, shift = pm.null_scenarios(X_gen, gam, inl)
        XG1 = g1_design(PS[ex].logit, E)
        full = ex in REGD.E5_FULL
        return Ds, E, inl, X_gen, gam, eta0, eta1, shift, XG1, full


    def e5_scenarios(XG1, eta0, eta1, shift, E, inl, full, targeting):
        if targeting == "none" or not full:
            return pm.build_scenarios(XG1, eta0, eta1, shift, E, inl, n_null=2000 if full else 1000, nulls=("N0", "N1"),
                                      power_ors=(), targeting=targeting)
        return pm.build_scenarios(XG1, eta0, eta1, shift, E, inl, n_null=2000, n_power=1000, targeting=targeting)


    def seed_tree():
        root = np.random.SeedSequence(REGD.SEED_A5)
        return {ex: dict(zip(REGD.E5_SCENARIOS, ch.spawn(len(REGD.E5_SCENARIOS))))
                for ex, ch in zip(REGD.E5_SETS, root.spawn(len(REGD.E5_SETS)))}


    # ------------------------------------------------------------------ R3
    if "R3" in PARTS:
        say("-" * 78); say("R3  A6 calibration constants (beta*, targeted and as-written G1 estimand ORs)")
        des = pd.read_csv(os.path.join(CAL, "e5cal_design.csv")).set_index("exposure")
        asw = pd.read_csv(os.path.join(CAL, "e5cal_design_as_written.csv")).set_index("exposure")
        mx = mxg = mxa = mxt = 0.0; n = 0
        for ex in REGD.E5_SETS:
            Ds, E, inl, X_gen, gam, eta0, eta1, shift, XG1, full = e5_design(ex)
            mxg = max(mxg, max(adiff(a, b) for a, b in zip(gam, des.loc[ex, [f"gen_{k}" for k in ("int", "logz", "loglen", "lognsnp", "L", "Lld")]])),
                      adiff(shift, des.loc[ex, "n2_shift"]))
            for sc in e5_scenarios(XG1, eta0, eta1, shift, E, inl, full, "estimand"):
                # the estimand achieved by the package's beta*, recomputed (not the requested target)
                th = pm.g1_target(XG1, pm.marginal_p(sc.eta, inl, sc.csd, sc.shift))
                mxt = max(mxt, adiff(th, sc.target_log_or))
                mx = max(mx, adiff(sc.beta, des.loc[ex, f"beta_{sc.name}"]), adiff(np.exp(th), des.loc[ex, f"target_OR_{sc.name}"]),
                         adiff(sc.target_or_as_written, des.loc[ex, f"asw_target_OR_{sc.name}"])); n += 1
            for sc in e5_scenarios(XG1, eta0, eta1, shift, E, inl, full, "none"):
                mxa = max(mxa, adiff(sc.target_or_as_written, asw.loc[ex, f"asw_target_OR_{sc.name}"])); n += 1
        check("R3", "generator coefficients and N2 shift", mxg, 1e-8, n=len(REGD.E5_SETS))
        check("R3", "achieved estimand minus target, recomputed from the package's beta*", mxt, 1e-8)
        check("R3", "beta*, achieved estimand OR, as-written OR vs the A6 run", mx, 1e-8)
        check("R3", "as-written implied ORs (A5 run)", mxa, 1e-8, n=n)

    # ------------------------------------------------------------------ R4 / R5
    RAWMAP19 = {"naive_fisher": ("fisher_logor", "fisher_p"), "G1_ps_firth": ("g1_logor", "g1_p"),
                "G1b_ps_mh": ("g1b_logor", "g1b_p"), "G1c_oe": ("g1c_logoe", "g1c_p"), "H_interaction": ("h_logor", "h_p"),
                "W1_cfl": ("w1_logor", "w1_p")}
    SC19 = [("N0", 2000), ("N1", 2000), ("N2", 2000), ("P-GW_OR1.5", 1000), ("P-GW_OR2.0", 1000), ("P-GW_OR2.7", 1000)]


    def e1_plasmode_design():
        Ds = DS["E1"]; E1 = Ds.E.values; inl = Ds.L.values; iid = Ds.interval.fillna(-1).astype(int).values
        lps = PS["E1"].logit
        _, psq = balance(Ds, "E", TIERP, PS["E1"].ps)
        X_gen = np.c_[np.ones(len(Ds)), Ds.x_z, Ds.x_len, Ds.x_snp, Ds.L.astype(float), Ds.ld]
        y = np.where(E1 == 0, Ds.Y.values, np.nan)
        gam = pm.fit_generator(X_gen, y, E1, maxit=MAXIT)
        eta0, eta1, shift = pm.null_scenarios(X_gen, gam, inl)
        XG1 = g1_design(lps, E1)
        scen = pm.build_scenarios(XG1, eta0, eta1, shift, E1, inl, targeting="none")
        design = pm.PlasmodeDesign(E=E1, X_g1=XG1, in_locus=inl, locus_id=iid, ps_quintile=np.asarray(psq),
                                   X_oe=np.c_[np.ones(len(Ds)), Ds[TIERP].values], X_h=h_design(lps, E1, inl),
                                   X_w1=np.c_[E1, Ds.ld.values, lps][inl], maxit=MAXIT, z_equiv=REGD.Z_STEP19)
        ss = np.random.SeedSequence(REGD.SEED_STEP19)
        seeds = {name: ss.spawn(nrep) for name, nrep in SC19}          # step 19: children spawned in turn
        return Ds, scen, design, seeds, gam, eta0, eta1, shift


    if "R4" in PARTS or "R4b" in PARTS or "R5" in PARTS:
        say("-" * 78); say(f"R4  plasmode replicates from the registered seeds (first {A.reps} per raw file)")
        Ds1, scen1, des1, seeds1, gam1, eta0_1, eta1_1, shift_1 = e1_plasmode_design()
        raw19 = {sc: pd.read_csv(os.path.join(A.step19, f"gw19_plasmode_raw_{sc}.csv")) for sc, _ in SC19}

    if "R4" in PARTS:
        mx = 0.0; n = 0; t1 = time.time()
        sub = [s for s in scen1]
        for s in sub:
            s.n_reps = A.reps
        R = pm.run(des1, sub, seeds1, estimators=("g1", "naive", "g1b", "g1c", "h", "w1", "equiv"), workers=A.workers)
        for sc, raw in raw19.items():
            rr = raw[raw.rep < A.reps]
            got = R[R.scenario == sc].set_index("rep")
            for test, (cl, cp) in RAWMAP19.items():
                w_ = rr[rr.test == test].set_index("rep")
                for k in range(A.reps):
                    mx = max(mx, adiff(got.loc[k, cl], w_.loc[k, "logor"]), adiff(got.loc[k, cp], w_.loc[k, "p"]))
                    n += 1
            w_ = rr[rr.test == "G1_equiv_upper_lt_2"].set_index("rep")
            if not len(w_):                                                # equivalence was simulated under N0-N2 only
                continue
            for k in range(A.reps):                                        # registered p = 0 <=> Wald upper < 2
                mx = max(mx, adiff(got.loc[k, "equiv_logor"], w_.loc[k, "logor"]),
                         adiff(1.0 - got.loc[k, "upper_lt_margin"], w_.loc[k, "p"]))
                n += 1
        check("R4", f"E1 plasmode, 6 scenarios x {A.reps} replicates x 7 tests ({(time.time() - t1) / 60:.1f} min)", mx, 1e-8, n=n)
        # E5, A6-targeted and as written
        mx = 0.0; me = 0; n = 0; t1 = time.time()
        for ex in REGD.E5_SETS:
            Ds, E, inl, X_gen, gam, eta0, eta1, shift, XG1, full = e5_design(ex)
            design = pm.PlasmodeDesign(E=E, X_g1=XG1, in_locus=inl, locus_id=Ds.interval.fillna(-1).astype(int).values,
                                       maxit=MAXIT)
            for tg, tag in (("estimand", ""), ("none", "asw_")):
                scs = e5_scenarios(XG1, eta0, eta1, shift, E, inl, full, tg)
                tree = seed_tree()                     # a fresh tree per run, as in the registered separate runs
                seeds = {s.name: list(tree[ex][s.name].spawn(s.n_reps)) for s in scs}
                for s in scs:
                    s.n_reps = A.reps
                R = pm.run(design, scs, seeds, estimators=("g1", "naive"), workers=A.workers)
                for s in scs:
                    raw = pd.read_csv(os.path.join(CAL, f"e5cal_{tag}raw_{ex}_{s.name}.csv")).set_index("rep")
                    got = R[R.scenario == s.name].set_index("rep")
                    for k in range(A.reps):
                        me += int(got.loc[k, "ysum"] != raw.loc[k, "ysum"]) + int(got.loc[k, "yexp"] != raw.loc[k, "yexp"])
                        for c in ("g1_logor", "g1_p", "g1_se", "fisher_logor", "fisher_p"):
                            mx = max(mx, adiff(got.loc[k, c], raw.loc[k, c]))
                        n += 1
        check("R4", f"E5 plasmode, 40 raw files x {A.reps} replicates: event-count mismatches", float(me), 0.0, n=n)
        check("R4", f"E5 plasmode, 40 raw files x {A.reps} replicates: G1 and Fisher ({(time.time() - t1) / 60:.1f} min)", mx, 1e-8, n=n)

    if "R4b" in PARTS:
        say("-" * 78); say("R4b D1 on the first 10 N0 replicates of the E1 plasmode (informational)")
        d1raw = pd.read_csv(os.path.join(A.step19, "d1_plasmode_raw_N0.csv")).set_index("rep")
        Ds = DS["E1"]; E1 = Ds.E.values
        B = aipw.basis(Ds, TIERP, LIN); Xd = Ds[TIERP].values.astype(float)
        sseeds = aipw.split_seeds(list(REGD.SEED_D1_E1), 50)
        n0 = scen1[0]; mismatch = 0; mxd = 0.0
        for k in range(10):
            s = k % 50
            fl = aipw.folds(E1, sseeds[s])
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                eh, _ = aipw.crossfit_e(B, Xd, E1, fl, sseeds[s])
            y, rng = pm.simulate(n0.eta, des1.in_locus, des1.locus_id, n0.csd, n0.shift, seeds1["N0"][k])
            th, p, w, ysum, yh = aipw.d1_replicate(y, rng, E1, B, Xd, fl, eh)
            mismatch += int(ysum != d1raw.loc[k, "ysum"]) + int(yh != d1raw.loc[k, "yhash"])
            mxd = max(mxd, adiff(th, d1raw.loc[k, "logrr"]))
        check("R4b", "outcome checksum and event-count mismatches (10 replicates)", float(mismatch), 0.0, n=10)
        check("R4b", "D1 ln RR, laptop run vs this environment", mxd, 0.0, info=True, n=10)

    if "R5" in PARTS:
        say("-" * 78); say("R5  plasmode summaries recomputed from the frozen raw files")
        # step 19
        rows = []
        for sc, raw in raw19.items():
            wide = raw.pivot(index="rep", columns="test", values=["logor", "p"])
            w = pd.DataFrame({"scenario": sc, "rep": wide.index.values})
            for test, (cl, cp) in RAWMAP19.items():
                w[cl] = wide[("logor", test)].values
                w[cp] = wide[("p", test)].values
            if ("p", "G1_equiv_upper_lt_2") in wide.columns:
                w["equiv_logor"] = wide[("logor", "G1_equiv_upper_lt_2")].values
                w["upper_lt_margin"] = (wide[("p", "G1_equiv_upper_lt_2")].values == 0).astype(float)
            rows.append(w)
        W19 = pd.concat(rows, ignore_index=True)
        S19 = pm.summarise(W19)
        F19 = pd.read_csv(os.path.join(A.step19, "gw19_plasmode.csv"))
        name = {"naive_fisher": "naive", "G1_ps_firth": "g1", "G1b_ps_mh": "g1b", "G1c_oe": "g1c", "H_interaction": "h",
                "W1_cfl": "w1", "G1_equiv_upper_lt_2": "g1_wald_upper_lt_margin"}
        mx = 0.0; n = 0
        for r in F19.itertuples():
            g = S19[(S19.scenario == r.scenario) & (S19.estimator == name[r.test])].iloc[0]
            mx = max(mx, adiff(g.reject, r.reject), adiff(g.failed, r.failed), adiff(g.mean_estimate, r.mean_logor),
                     adiff(g.mc_se, r.mc_se))
            n += 1
        check("R5", "gw19_plasmode.csv from the raw files (E1 equivalence as P(upper < 2))", mx, 1e-12, n=n)
        # step 20 (A6 and as written)
        for tag, fn in (("", "e5cal_plasmode.csv"), ("asw_", "e5cal_plasmode_as_written.csv")):
            F = pd.read_csv(os.path.join(CAL, fn)); mx = 0.0; n = 0
            for ex, Fx in F.groupby("exposure", sort=False):
                files = [f"e5cal_{tag}raw_{ex}_{sc}.csv" for sc in Fx.scenario.unique()]    # exact names (E5c vs E5c_naive)
                raws = pd.concat([pd.read_csv(os.path.join(CAL, f)) for f in files])
                tg = dict(zip(Fx[Fx.test == "G1_ps_firth"].scenario, Fx[Fx.test == "G1_ps_firth"].target_logor))
                Sx = pm.summarise(raws, targets=tg)
                for r in Fx.itertuples():
                    g = Sx[(Sx.scenario == r.scenario) & (Sx.estimator == ("g1" if r.test == "G1_ps_firth" else "naive"))].iloc[0]
                    d = [adiff(g.reject, r.reject), adiff(g.mc_se, r.mc_se), adiff(g.failed, r.failed), adiff(g.mean_estimate, r.mean_logor)]
                    if r.test == "G1_ps_firth":
                        d += [adiff(g.lower95_gt1, r.lower95_gt1), adiff(g.lower975_gt1, r.lower975_gt1),
                              adiff(g.upper95_lt_margin, r.upper95_lt2), adiff(g.upper975_lt_margin, r.upper975_lt2),
                              adiff(g.mean_minus_target, r.mean_minus_target)]
                        d += [adiff(g[f"cls_{k}"], getattr(r, f"cls_{k}")) for k in ("enriched", "modest", "no_material", "inconclusive", "failed")]
                    mx = max(mx, max(d)); n += 1
                    if isinstance(r.verdict, str) and r.verdict and g.verdict != r.verdict:
                        mx = max(mx, 1.0)
            check("R5", f"{fn} from the raw files", mx, 1e-12, n=n)

    # ------------------------------------------------------------------ R6
    if "R6" in PARTS:
        say("-" * 78); say(f"R6  matched negative-control sets from the registered seeds (first {A.sets_n})")
        tree = seed_tree()
        for ex, fn, rng in (("E5a", "e5cal_nc_E5a.csv", np.random.default_rng(tree["E5a"]["NC"])),
                            ("E5c", "e5cal_nc_E5c.csv", np.random.default_rng(tree["E5c"]["NC"])),
                            ("E1", "gw19_e4_sets.csv", np.random.default_rng(list(REGD.SEED_E4)))):
            Ds = DS[ex].copy()
            Ds["Y"] = np.where(Ds.E.values == 0, Ds.Y.values, np.nan)        # exposed outcomes never read
            t1 = time.time()
            R = negative_control.run(Ds, "Y", "E", TIERP, LIN, n_sets=A.sets_n, rng=rng, ps=PS[ex].ps,
                                     workers=A.workers, maxit=MAXIT)
            F = pd.read_csv(os.path.join(CAL if ex != "E1" else A.step19, fn)).iloc[:A.sets_n]
            ev = int((R.events.values != F.events.values).sum())
            d = max(vdiff(R.logor.values, F.logor.values), vdiff(R.p.values, F.p.values))
            info = ex == "E1"
            check("R6", f"{ex} sets: event-count mismatches ({(time.time() - t1) / 60:.1f} min)", float(ev), 0.0, info=info, n=A.sets_n)
            check("R6", f"{ex} sets: ln OR and P", float(d), 1e-8, info=info, n=A.sets_n)
            if info and ev:
                F_all = pd.read_csv(os.path.join(A.step19, fn))
                say(f"      E4 draws differ from the laptop run; frozen rate {negative_control.summarise(F_all).reject:.3f}")

    # ------------------------------------------------------------------ R7
    if "R7" in PARTS:
        say("-" * 78); say("R7  D1 on the real outcome (50 splits)")
        t1 = time.time()
        Ds = DS["E5a"]
        r = aipw.d1(Ds, "Y", "E", TIERP, LIN, entropy=[REGD.SEED_A5, 1, 1], workers=A.workers)
        w_ = E5R[(E5R.exposure == "E5a") & E5R.analysis.str.startswith("D1")].iloc[0]
        d = max(dlog(r.rr, w_.OR), dlog(r.lo, w_.lo), dlog(r.hi, w_.hi))
        check("R7", f"E5a D1 RR and bounds, log scale ({(time.time() - t1) / 60:.1f} min)", d, 1e-6)
        t1 = time.time()
        r = aipw.d1(DS["E1"], "Y", "E", TIERP, LIN, entropy=list(REGD.SEED_D1_E1), z=REGD.Z_STEP19, workers=A.workers)
        w_ = pd.read_csv(os.path.join(A.step19, "d1_results.csv")).set_index("exposure").loc["E1"]
        d = max(dlog(r.rr, w_.RR_ATT), dlog(r.lo, w_.lo), dlog(r.hi, w_.hi))
        check("R7b", f"E1 D1 (laptop run) RR and bounds, log scale ({(time.time() - t1) / 60:.1f} min)", d, 0.0, info=True)

    # ------------------------------------------------------------------ R8
    if "R8" in PARTS:
        say("-" * 78); say("R8  decision functions on the registered estimates")
        mism = 0; n = 0
        parse = lambda s: [float(v) for v in str(s).strip("[]").split()]
        for ex in REGD.E5_COPRIMARY:
            d = E5D.loc[ex]
            lo, hi = parse(d.wald975); plo, phi = parse(d.prof975)
            mism += decisions.four_way_rule(lo, hi) != d.class_wald975
            mism += decisions.four_way_rule(plo, phi) != d.class_prof975
            cal = pd.read_csv(os.path.join(CAL, "e5cal_design.csv")).set_index("exposure").loc[ex]
            reading = decisions.coprimary_reading(lo, hi, bool(cal.g1_calibrated), bool(cal.nc_passes), gate="pass")
            want = d.final if not str(d.final).startswith("NOT INTERPRETED") else "not interpreted"
            mism += reading != want; n += 3
        lo, hi = parse(E5D.loc["E5g", "wald95"])
        dg = pd.read_csv(os.path.join(CAL, "e5cal_design.csv")).set_index("exposure").loc["E5g"]
        gate = decisions.gate_rule(lo, power=float(dg.gate_power_or27), calibrated=bool(dg.g1_calibrated))
        mism += (gate != "uninformative") != (dg.gate_evaluable in (True, "True"))
        mism += gate != ("pass" if E5D.loc["E5g", "gate_pass"] in (True, "True") else "fail"); n += 2
        mism += decisions.network_rule(G19.loc["E1 G1  PS-adjusted Firth  <-- PRIMARY", "hi"]) != "inconclusive"
        mism += decisions.network_rule(G19.loc["E2 G1  PS-adjusted Firth", "hi"]) != "doubling excluded"; n += 2
        F19 = pd.read_csv(os.path.join(A.step19, "gw19_plasmode.csv"))
        t = F19[F19.test == "G1_ps_firth"].set_index("scenario").reject
        mism += decisions.validity_rule(t["N0"], t["N1"]) is not True; n += 1
        nc = pd.read_csv(os.path.join(CAL, "e5cal_nc.csv")).set_index("exposure")
        mism += decisions.band_rule(G19.loc["E4 rejection rate at P<0.05", "OR"]) is not True
        mism += decisions.band_rule(nc.loc["E5a", "reject"]) is not True
        mism += decisions.band_rule(nc.loc["E5c", "reject"]) is not False; n += 3
        check("R8", "readings that differ from the registered ones", float(mism), 0.0, n=n)

    # ------------------------------------------------------------------ R9
    if "R9" in PARTS:
        say("-" * 78); say("R9  convergence flag: registered cap (100) vs package default (1,000)")
        flags100, flags_def, dmax = {}, {}, 0.0
        for ex in ORDER:
            y, e, lps = DS[ex].Y.values, DS[ex].E.values, PS[ex].logit
            X = g1_design(lps, e)
            f100 = nb.firth_fit(X, y, maxit=100, warn=False); fd = nb.firth_fit(X, y, warn=False)
            flags100[ex], flags_def[ex] = f100.converged, fd.converged
            dmax = max(dmax, adiff(f100.beta[1], fd.beta[1]))
        ok_flags = all(flags_def.values()) and (not flags100["E1"]) and all(v for k, v in flags100.items() if k != "E1")
        check("R9", "flags as expected (default: all converged; maxit 100: only E1 not)", 0.0 if ok_flags else 1.0, 0.0, n=10)
        check("R9", "ln OR change between maxit 100 and the default", dmax, 1e-7, n=10)
        say("      maxit 100 converged: " + ", ".join(f"{k} {v}" for k, v in flags100.items()))

    # ------------------------------------------------------------------ R10 (post-spec)
    if "R10" in PARTS:
        say("-" * 78)
        say("R10 POST-SPEC (added after the step-29 spec): amendment A10 outputs regenerated from their seeds")
        S10 = dict(zip(["N0", "N1", "N2", "P1.5", "P2.0", "NC_E2", "BOOT5", "BOOT2", "BOOT10"],
                       np.random.SeedSequence(REGD.SEED_A10).spawn(9)))
        # A10.1: E2-design estimand-targeted plasmode (generator on the non-E2 genes; G1 per replicate; profile
        # interval under N0 and P2.0 only, as registered)
        Ds = DS["E2"]; E = Ds.E.values; inl = Ds.L.values; iid = Ds.interval.fillna(-1).astype(int).values
        X_gen = np.c_[np.ones(len(Ds)), Ds.x_z, Ds.x_len, Ds.x_snp, Ds.L.astype(float), Ds.ld]
        gam = pm.fit_generator(X_gen, np.where(E == 0, Ds.Y.values, np.nan), E, maxit=MAXIT)
        eta0, eta1, shift = pm.null_scenarios(X_gen, gam, inl)
        XG1 = g1_design(PS["E2"].logit, E)
        scen = pm.build_scenarios(XG1, eta0, eta1, shift, E, inl, n_null=2000, n_power=1000, power_ors=(1.5, 2.0),
                                  targeting="estimand", power_names={1.5: "P1.5", 2.0: "P2.0"})
        d10 = pd.read_csv(os.path.join(A.step25, "a10_e2_plasmode_design.csv")).set_index("scenario")
        mx = 0.0
        for s in scen:
            th = pm.g1_target(XG1, pm.marginal_p(s.eta, inl, s.csd, s.shift))
            mx = max(mx, adiff(s.beta, d10.loc[s.name, "beta_star"]), adiff(th, np.log(d10.loc[s.name, "target_OR"])),
                     adiff(s.target_or_as_written, d10.loc[s.name, "as_written_target_OR"]))
        check("R10", "E2 design: beta*, achieved and as-written estimands vs A10", mx, 1e-8, n=len(scen))
        seeds = {s.name: list(S10[s.name].spawn(s.n_reps)) for s in scen}          # spawned once, reused below
        for s in scen:
            s.n_reps = A.reps
        des = pm.PlasmodeDesign(E=E, X_g1=XG1, in_locus=inl, locus_id=iid, maxit=MAXIT)
        t1 = time.time()
        R = pm.run(des, scen, seeds, estimators=("g1",), workers=A.workers)
        Rp = pm.run(des, [s for s in scen if s.name in ("N0", "P2.0")], seeds, estimators=("g1_profile",),
                    workers=A.workers)
        mx = 0.0; me = 0; n = 0
        for s in scen:
            raw = pd.read_csv(os.path.join(A.step25, f"a10_e2_plasmode_raw_{s.name}.csv")).set_index("rep")
            got = R[R.scenario == s.name].set_index("rep")
            gp = Rp[Rp.scenario == s.name].set_index("rep")
            for k in range(A.reps):
                me += int(got.loc[k, "ysum"] != raw.loc[k, "ysum"]) + int(got.loc[k, "yexp"] != raw.loc[k, "yexp"])
                mx = max(mx, adiff(got.loc[k, "g1_logor"], raw.loc[k, "logor"]), adiff(got.loc[k, "g1_p"], raw.loc[k, "p"]),
                         adiff(got.loc[k, "g1_se"], raw.loc[k, "se"]))
                if len(gp):
                    mx = max(mx, adiff(gp.loc[k, "g1_prof_lo"], raw.loc[k, "prof_lo"]),
                             adiff(gp.loc[k, "g1_prof_hi"], raw.loc[k, "prof_hi"]))
                n += 1
        check("R10", f"E2 plasmode, 5 scenarios x {A.reps}: event-count mismatches", float(me), 0.0, n=n)
        check("R10", f"E2 plasmode: G1 ln OR, P, SE and profile bounds ({(time.time() - t1) / 60:.1f} min)", mx, 1e-8, n=n)
        # A10.2: E2-matched negative control
        t1 = time.time()
        D0 = Ds.copy()
        D0["Y"] = np.where(E == 0, Ds.Y.values, np.nan)                       # exposed outcomes never read
        Rn = negative_control.run(D0, "Y", "E", TIERP, LIN, n_sets=A.sets_n, rng=np.random.default_rng(S10["NC_E2"]),
                                  ps=PS["E2"].ps, workers=A.workers, maxit=MAXIT)
        F = pd.read_csv(os.path.join(A.step25, "a10_nc_e2.csv")).iloc[:A.sets_n]
        check("R10", f"E2 matched control: event-count mismatches ({(time.time() - t1) / 60:.1f} min)",
              float((Rn.events.values != F.events.values).sum()), 0.0, n=A.sets_n)
        check("R10", "E2 matched control: ln OR and P", max(vdiff(Rn.logor.values, F.logor.values),
                                                              vdiff(Rn.p.values, F.p.values)), 1e-8, n=A.sets_n)
        # A10.3: 5-Mb genomic-block bootstrap of E1 and E2. Resamples repeat genes, so tied logit PS values can be
        # split by one ulp on another platform and move a spline knot: the tolerance is 1e-3 on ln OR.
        from nomibench import bootstrap
        t1 = time.time()
        Bt = bootstrap.run(DS["E1"], "Y", ["E1", "E2"], TIERP, LIN, chrom="chr_use", tss="tss_use", block_size=5e6,
                           seeds=list(S10["BOOT5"].spawn(2000))[:A.reps], workers=A.workers, maxit=MAXIT)
        F = pd.read_csv(os.path.join(A.step25, "a10_boot_5Mb.csv")).iloc[:A.reps]
        check("R10", f"5-Mb bootstrap: gene and event count mismatches ({(time.time() - t1) / 60:.1f} min)",
              float((Bt.n_genes.values != F.n_genes.values).sum() + (Bt.events.values != F.events.values).sum()),
              0.0, n=A.reps)
        check("R10", "5-Mb bootstrap: ln OR of E1 and E2", max(vdiff(Bt.lnOR_E1.values, F.lnOR_E1.values),
                                                             vdiff(Bt.lnOR_E2.values, F.lnOR_E2.values)), 1e-3, n=A.reps)

    # ------------------------------------------------------------------ write
    C = pd.DataFrame(CHECKS)
    C.to_csv(os.path.join(A.out, "repro_checks.csv"), index=False)
    fails = C[C.result == "FAIL"]
    say("=" * 78)
    say(f"{(C.result == 'PASS').sum()} passed, {len(fails)} failed, {(C.result == 'info').sum()} informational "
        f"| {(time.time() - t0) / 60:.1f} min")
    for r in fails.itertuples():
        say(f"  FAILED: {r.part} {r.check} ({r.max_abs_diff:.3g} > {r.tol:g})")
    json.dump(dict(nomibench=nb.__version__, date=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                   versions={m: __import__(m).__version__ for m in ("numpy", "scipy", "pandas", "statsmodels", "patsy", "sklearn")},
                   parts=sorted(PARTS), checks=CHECKS), open(os.path.join(A.out, "repro_summary.json"), "w"), indent=1)


if __name__ == "__main__":      # required for multiprocessing on Windows (spawn)
    main()
