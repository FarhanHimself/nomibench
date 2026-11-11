#!/usr/bin/env python3
"""
build_covariates.py -- the U_gw covariate table of paper 1 (spec v2; amendment A4 section 10.3).

A port of the registered script 19a_covariates.py (sha256 e132349c...) with input paths as arguments. The logic is
unchanged. NO OUTCOME IS READ: from the SMR table only the eQTL-side columns are loaded (probe, position,
b_eQTL, se_eQTL).

MetaBrain-derived output: do not redistribute it.

  python build_covariates.py --smr smr_gw_all.csv --topeffects 2021-07-23-cortex-EUR-80PCs-TopEffects.txt.gz \
      --gencode gencode.v42.basic.annotation.gtf.gz --table-s3 41588_2026_2583_MOESM4_ESM.xlsx \
      --lrn lrn_causal_genes.csv --loeuf gnomad_v4.1.1_loeuf.csv --gtex gtex_v11_ba9_tau.csv \
      --out ugw_covariates.csv [--check frozen_ugw_covariates.csv]

Inputs:
  --smr         SMR output for every MetaBrain cortex-EUR eGene against the AD GWAS (see README, step 2)
  --topeffects  MetaBrain cortex-EUR TopEffects file (per-gene meta-analysis N, datasets, SNPs tested)
  --gencode     GENCODE v42 basic annotation (gene coordinates, type, name)
  --table-s3    the 2026 consensus AD GWAS supplementary Table S3 (locus intervals and index variants)
  --lrn         RADC LRN gene table with columns gene, ensg, causal_to_pheno, n_edges (exposures E1, E2)
  --loeuf       gnomAD v4.1.1 LOEUF per gene (columns ensg, gnomad_loeuf)            [tier S2 only]
  --gtex        GTEx v11 BA9 log2 TPM and tissue specificity tau (ensg, gtex_ba9_log2tpm, gtex_tau) [tier S2 only]
"""
import argparse
import gzip
import re

import numpy as np
import pandas as pd

GVC = ["ABCA7", "ABI3", "APOE", "APP", "BIN1", "CASP7", "PLCG2", "PSEN1", "PSEN2", "SORL1", "SPI1", "TREM2"]
"""E3: the 12 genes linked to the ADSP Gene Verification Committee's Phase-1 high-confidence loci."""


def gencode(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ln[0] == "#":
                continue
            p = ln.split("\t", 8)
            if p[2] != "gene":
                continue
            a = p[8]
            rows.append((re.search(r'gene_id "([^"]+)"', a).group(1), p[0].replace("chr", ""), int(p[3]), int(p[4]), p[6],
                         re.search(r'gene_type "([^"]+)"', a).group(1), re.search(r'gene_name "([^"]+)"', a).group(1)))
    G = pd.DataFrame(rows, columns=["gid", "chr", "start", "end", "strand", "gtype", "gname"])
    G = G[~G.gid.str.contains("_PAR_")].copy()
    G["ensg"] = G.gid.str.split(".").str[0]
    G = G.drop_duplicates("ensg")
    G["tss"] = np.where(G.strand == "+", G.start, G.end)
    G["glen"] = G.end - G.start + 1
    pc = {c: np.sort(s.tss.values) for c, s in G[G.gtype == "protein_coding"].groupby("chr")}

    def dens(chrom, tss):
        t = pc.get(chrom)
        if t is None or pd.isna(tss):
            return np.nan
        return np.searchsorted(t, tss + 1_000_000, "right") - np.searchsorted(t, tss - 1_000_000, "left")

    G["dens1mb"] = [dens(c, t) for c, t in zip(G.chr, G.tss)]
    return G


def loci(path):
    s3 = pd.read_excel(path, sheet_name="Table S3", skiprows=3)
    s3 = s3[pd.to_numeric(s3["Locus number"], errors="coerce").notna()].copy()
    s3["chr"] = s3["chr"].astype(int).astype(str)
    s3["locus"] = s3["Locus number"].astype(int)
    LOC = (s3[["locus", "chr", "start", "end", "Locus name"]].drop_duplicates(["locus", "start", "end"])
           .rename(columns={"start": "lstart", "end": "lend", "Locus name": "lname"}).reset_index(drop=True))
    LOC["interval"] = LOC.index
    IDX = s3[["locus", "position"]].rename(columns={"position": "idx_pos"})
    return LOC, IDX


def assign_all(pos, chrom, LOC, IDX):
    """The interval containing the position; D = distance to that interval's nearest index variant; if several
    intervals contain it, the one with the nearest index variant."""
    out_i = np.full(len(pos), np.nan)
    out_d = np.full(len(pos), np.nan)
    for k, (p_, c_) in enumerate(zip(pos, chrom)):
        if pd.isna(p_) or pd.isna(c_):
            continue
        hit = LOC[(LOC.chr == str(c_)) & (LOC.lstart <= p_) & (LOC.lend >= p_)]
        best = (np.nan, np.nan)
        for _, h in hit.iterrows():
            ix = IDX[(IDX.locus == h.locus) & (IDX.idx_pos >= h.lstart) & (IDX.idx_pos <= h.lend)].idx_pos
            d = (ix - p_).abs().min()
            if np.isnan(best[1]) or d < best[1]:
                best = (h.interval, d)
        out_i[k], out_d[k] = best
    return out_i, out_d


def build(smr_path, topeffects, G, LOC, IDX):
    """Tier P and loci for every gene in an SMR output file (eQTL-side columns only)."""
    S = pd.read_csv(smr_path, usecols=["probeID", "Gene", "ProbeChr", "Probe_bp", "b_eQTL", "se_eQTL"])
    S["z_e"] = (S.b_eQTL / S.se_eQTL).abs()
    S["ProbeChr"] = S.ProbeChr.astype(str)
    S = S.sort_values("z_e", ascending=False).drop_duplicates("probeID").rename(columns={"probeID": "ensg", "Gene": "symbol"})
    T = pd.read_csv(topeffects, sep="\t", usecols=["Gene", "MetaP", "MetaPN", "NrDatasets", "NrTestedSNPs"])
    T["ensg"] = T.Gene.str.replace(r"\.\d+$", "", regex=True)
    T = T.drop(columns="Gene").drop_duplicates("ensg").rename(columns={
        "MetaP": "mb_top_p", "MetaPN": "mb_N", "NrDatasets": "mb_ndata", "NrTestedSNPs": "mb_nsnp"})
    X = (S[["ensg", "symbol", "ProbeChr", "Probe_bp", "z_e"]].merge(T, on="ensg", how="left")
         .merge(G[["ensg", "chr", "tss", "glen", "dens1mb", "gtype", "gname"]], on="ensg", how="left"))
    X["chr_use"] = X.chr.fillna(X.ProbeChr)
    X["tss_use"] = X.tss.fillna(X.Probe_bp)
    X["interval"], X["D"] = assign_all(X.tss_use.values, X.chr_use.values, LOC, IDX)
    X["L"] = X.interval.notna()
    X["ld"] = np.where(X.L, np.log10(X.D.fillna(0) + 1000), 0.0)
    return X


def main():
    ap = argparse.ArgumentParser()
    for k in ("smr", "topeffects", "gencode", "table-s3", "lrn", "loeuf", "gtex", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--check", default=None, help="a frozen covariate table to compare against")
    a = ap.parse_args()
    G = gencode(a.gencode)
    LOC, IDX = loci(a.table_s3)
    X = build(a.smr, a.topeffects, G, LOC, IDX)
    L = pd.read_csv(a.lrn)[["gene", "ensg", "causal_to_pheno", "n_edges"]]
    assert L.ensg.is_unique
    X = X.merge(L[["ensg", "causal_to_pheno", "n_edges"]], on="ensg", how="left")
    X["E2"] = X.causal_to_pheno.notna()
    X["E1"] = X.causal_to_pheno.fillna(0).astype(int) == 1
    gv = G[G.gname.isin(GVC) & (G.gtype == "protein_coding")][["gname", "ensg"]]
    assert gv.gname.nunique() == len(GVC) and len(gv) == len(GVC), gv
    X["E3"] = X.ensg.isin(gv.ensg)
    X = X.merge(pd.read_csv(a.loeuf), on="ensg", how="left")
    X = X.merge(pd.read_csv(a.gtex), on="ensg", how="left")
    X["x_N"] = np.log10(X.mb_N)
    X["x_snp"] = np.log10(X.mb_nsnp)
    X["x_len"] = np.log10(X.glen)
    X["x_den"] = X.dens1mb.astype(float)
    X["x_z"] = np.log(X.z_e)
    X["x_nd"] = X.mb_ndata.astype(float)
    X["tierP_complete"] = X[["mb_N", "mb_nsnp", "mb_ndata", "glen", "dens1mb", "z_e"]].notna().all(axis=1)
    mhc = (X.chr_use.astype(str) == "6") & X.tss_use.between(28_500_000, 33_500_000)
    apoe = (X.chr_use.astype(str) == "19") & X.tss_use.between(44_000_000, 46_500_000)
    X["mhc_apoe"] = mhc | apoe
    X.to_csv(a.out, index=False, float_format="%.17g", lineterminator="\n")   # same bytes on every platform
    print(f"U_gw genes {len(X):,} | complete Tier P {int(X.tierP_complete.sum()):,} | in-locus {int(X.L.sum())} in "
          f"{X.interval.nunique()} intervals | MHC/APOE {int(X.mhc_apoe.sum())} | E1 {int(X.E1.sum())} | "
          f"E2 {int(X.E2.sum())} | E3 {int(X.E3.sum())} of 12")
    if a.check:
        F = pd.read_csv(a.check, dtype={"ensg": str})
        m = X.merge(F, on="ensg", suffixes=("", "_frozen"))
        cols = ["x_N", "x_snp", "x_nd", "x_len", "x_den", "x_z", "L", "ld", "interval", "E1", "E2", "E3", "mhc_apoe"]
        bad = {c: int((~np.isclose(m[c].astype(float), m[c + "_frozen"].astype(float), rtol=0, atol=1e-12,
                                    equal_nan=True)).sum()) for c in cols}
        print(f"check: {len(m):,} of {len(F):,} frozen genes matched; non-identical values per column: {bad}")


if __name__ == "__main__":
    main()
