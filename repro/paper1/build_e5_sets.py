#!/usr/bin/env python3
"""
20a_e5_exposures.py  --  build the E5 external nomination sets (METHOD_SPEC_v2, amendment A5).

EXPOSURE SIDE ONLY. This script never reads Y_gw or any SMR result. The optional
--ugw file is used only to report how many genes of each set are testable
(present in U_gw); outcome values are never touched.

Inputs (pinned in A5):
  --agora   agora_nominated_targets_syn13363290_v119_2026-09-28.csv
  --ot      opentargets_26.09_AD_MONDO_0004975_by_datatype_direct_wide.csv
Optional:
  --ugw     a file of U_gw Ensembl gene IDs (one per line, or a column 'ensembl_id'/'gene_id')

Outputs:
  <out>/e5_exposure_sets_A5.tsv   one row per gene in any source, with 0/1 set flags
  <out>/e5_exposure_summary_A5.txt

Set definitions (A5 §2). OT genes are tiered by their STRONGEST NON-GENETIC evidence;
genetic evidence never enters an OT exposure definition (that would condition the
exposure on a proxy of the outcome):
  E5a  Agora nominated, input_data contains none of {Genetics, Risk Scores, Unknown}   [co-primary]
  E5b  Agora nominated, input_data contains Genetics                                   [secondary]
  E5c  OT 26.09 direct AD evidence from curated knowledge:
       known_drug OR affected_pathway OR animal_model                                  [co-primary]
  E5e  OT rna_expression (differential expression), not in E5c                        [secondary]
  E5d  OT literature (Europe PMC) as the ONLY non-genetic evidence                    [secondary]
  E5g  OT genetic datatype: genetic_association, genetic_literature,
       somatic_mutation                                                                [positive reference, gate]
  E5c_naive  E5c within the stratum of genes NOT in E5g (stratum flag = not E5g)      [secondary]
"""
import argparse
import os
import re
import sys

import pandas as pd

AGORA_GENETIC_INPUTS = {"Genetics", "Risk Scores", "Unknown"}
OT_CURATED = ["known_drug", "affected_pathway", "animal_model"]
OT_EXPRESSION = ["rna_expression"]
OT_GENETIC = ["genetic_association", "genetic_literature", "somatic_mutation"]
OT_LITERATURE = "literature"
ENSG = re.compile(r"^ENSG\d{11}$")


def norm_id(x: str) -> str:
    return str(x).strip().split(".")[0].upper()


def read_id_list(path: str) -> set:
    """U_gw gene list: one ID per line, or a delimited file (tab/comma) with a column named
    ensembl_id / gene_id / ensembl / gene / probe. No delimiter sniffing (unreliable on one column)."""
    with open(path, encoding="utf-8-sig") as fh:
        rows = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
    if not rows:
        sys.exit(f"[ugw] {path} is empty")
    delim = "\t" if "\t" in rows[0] else ("," if "," in rows[0] else None)
    split = (lambda s: [t.strip().strip('"') for t in s.split(delim)]) if delim else (lambda s: [s.strip().strip('"')])
    head = split(rows[0])
    if any(ENSG.match(norm_id(t)) for t in head):  # no header row
        col, body = next(i for i, t in enumerate(head) if ENSG.match(norm_id(t))), rows
    else:
        names = [h.lower() for h in head]
        hits = [i for i, h in enumerate(names) if h in ("ensembl_id", "gene_id", "ensembl", "gene", "probe")]
        if not hits:
            sys.exit(f"[ugw] no Ensembl ID column in header {head}")
        col, body = hits[0], rows[1:]
    ids = {norm_id(split(r)[col]) for r in body}
    bad = [i for i in ids if not ENSG.match(i)]
    if bad:
        sys.exit(f"[ugw] {len(bad)} non-Ensembl entries, e.g. {bad[:3]}")
    return ids


def load_agora(path: str) -> pd.DataFrame:
    a = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    need = {"ensembl_gene_id", "hgnc_symbol", "input_data"}
    miss = need - set(a.columns)
    if miss:
        sys.exit(f"Agora file lacks columns: {miss}")
    a["ensembl_id"] = a["ensembl_gene_id"].map(norm_id)
    a["inputs"] = a["input_data"].map(lambda s: {t.strip() for t in s.split(";") if t.strip()})
    bad = ~a["ensembl_id"].str.match(ENSG)
    if bad.any():
        print(f"[agora] WARNING {bad.sum()} rows with malformed Ensembl IDs dropped", file=sys.stderr)
        a = a[~bad]
    if a["ensembl_id"].duplicated().any():
        sys.exit("[agora] duplicated Ensembl IDs -- resolve before building sets")
    a["E5a"] = a["inputs"].map(lambda s: int(len(s & AGORA_GENETIC_INPUTS) == 0))
    a["E5b"] = a["inputs"].map(lambda s: int("Genetics" in s))
    a["agora_other"] = ((a["E5a"] == 0) & (a["E5b"] == 0)).astype(int)  # Risk Scores / Unknown without Genetics
    a["agora_nominated"] = 1
    return a[["ensembl_id", "hgnc_symbol", "agora_nominated", "E5a", "E5b", "agora_other", "input_data"]]


def load_ot(path: str) -> pd.DataFrame:
    o = pd.read_csv(path, encoding="utf-8-sig")
    if "targetId" not in o.columns:
        sys.exit("OT file lacks targetId")
    o["ensembl_id"] = o["targetId"].map(norm_id)
    if o["ensembl_id"].duplicated().any():
        sys.exit("[ot] duplicated target IDs")
    present = {}
    for c in o.columns:
        if c.startswith("score_"):
            dt = c[len("score_"):]
            present[dt] = (o[c].fillna(0) > 0).astype(int)
    unknown = set(present) - set(OT_CURATED) - set(OT_EXPRESSION) - set(OT_GENETIC) - {OT_LITERATURE}
    if unknown:  # fail closed: a datatype A5 has not classified cannot be used
        sys.exit(f"[ot] unclassified datatypes {sorted(unknown)} -- A5 must classify them before use")
    P = pd.DataFrame(present, index=o.index)
    for dt in OT_CURATED + OT_EXPRESSION + OT_GENETIC + [OT_LITERATURE]:
        if dt not in P:
            P[dt] = 0
    out = pd.DataFrame({"ensembl_id": o["ensembl_id"],
                        "ot_symbol": o.get("symbol_gencode_v38", pd.Series([""] * len(o)))})
    out["ot_any"] = 1
    curated = P[OT_CURATED].sum(axis=1) > 0
    expression = P[OT_EXPRESSION].sum(axis=1) > 0
    literature = P[OT_LITERATURE] > 0
    out["E5c"] = curated.astype(int)
    out["E5e"] = (expression & ~curated).astype(int)
    out["E5d"] = (literature & ~curated & ~expression).astype(int)
    out["E5g"] = (P[OT_GENETIC].sum(axis=1) > 0).astype(int)
    for dt in P.columns:
        out[f"ot_{dt}"] = P[dt]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agora", required=True)
    ap.add_argument("--ot", required=True)
    ap.add_argument("--ugw", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ag = load_agora(a.agora)
    ot = load_ot(a.ot)
    m = ag.merge(ot, on="ensembl_id", how="outer")
    flag_cols = ["agora_nominated", "E5a", "E5b", "agora_other", "ot_any", "E5c", "E5e", "E5d", "E5g"] + \
                [c for c in m.columns if c.startswith("ot_") and c not in ("ot_any", "ot_symbol")]
    for c in flag_cols:
        m[c] = m[c].fillna(0).astype(int)
    m["symbol"] = m["hgnc_symbol"].where(m["hgnc_symbol"].notna() & (m["hgnc_symbol"] != ""), m["ot_symbol"])
    m["E5c_naive_stratum"] = (m["E5g"] == 0).astype(int)
    m["E5c_naive"] = ((m["E5c"] == 1) & (m["E5g"] == 0)).astype(int)
    cols = ["ensembl_id", "symbol", "agora_nominated", "E5a", "E5b", "agora_other",
            "ot_any", "E5c", "E5e", "E5d", "E5g", "E5c_naive", "E5c_naive_stratum"] + \
           sorted(c for c in m.columns if c.startswith("ot_") and c not in ("ot_any", "ot_symbol")) + ["input_data"]
    m = m[cols].sort_values("ensembl_id")
    tsv = os.path.join(a.out, "e5_exposure_sets_A5.tsv")
    m.to_csv(tsv, sep="\t", index=False, lineterminator="\n")  # LF on every OS, so the sha256 is portable

    lines = []
    say = lambda s="": (lines.append(s), print(s))
    say("E5 exposure sets (amendment A5) -- exposure side only, no outcome read")
    say(f"Agora nominated: {int(m.agora_nominated.sum())} | OT genes with direct AD evidence: {int(m.ot_any.sum())}")
    sets = ["E5a", "E5b", "agora_other", "E5c", "E5e", "E5d", "E5g", "E5c_naive"]
    for s in sets:
        say(f"  {s:<12} {int(m[s].sum()):>6}")
    say("Pairwise overlaps (n, Jaccard):")
    core = ["E5a", "E5b", "E5c", "E5e", "E5d", "E5g"]
    for i, s in enumerate(core):
        for t in core[i + 1:]:
            x, y = m[s] == 1, m[t] == 1
            inter, union = int((x & y).sum()), int((x | y).sum())
            say(f"  {s} & {t}: {inter:>5}  J={inter / union if union else 0:.3f}")

    if a.ugw:
        ids = read_id_list(a.ugw)
        say(f"\nTestability against U_gw ({len(ids)} genes; membership only, no outcome):")
        for s in sets:
            n = int(m[s].sum())
            k = int(((m[s] == 1) & m["ensembl_id"].isin(ids)).sum())
            say(f"  {s:<12} {k:>6} of {n:<6} testable ({(k / n if n else 0):.1%})")

    with open(os.path.join(a.out, "e5_exposure_summary_A5.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {tsv}")


if __name__ == "__main__":
    main()
