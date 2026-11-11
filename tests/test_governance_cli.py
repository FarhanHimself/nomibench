"""Release hygiene: no private data or paths in the distributed files; the command line works."""
import os
import re
import subprocess
import sys

import pytest

from conftest import make_genes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = ("src", "tests", "docs", "repro", "examples", "README.md", "CITATION.cff", ".zenodo.json", "pyproject.toml")
PATTERNS = {
    "Ensembl gene ID": re.compile(r"ENSG\d{11}"),
    "private upload path": re.compile(r"/mnt/user-data|/home/claude"),
    "Windows research-data path": re.compile(r"[A-Za-z]:[\\/]+Research|Research Data[\\/]"),
}


def _files():
    for item in SCAN:
        p = os.path.join(ROOT, item)
        if os.path.isfile(p):
            yield p
        for d, _, fs in os.walk(p):
            for f in fs:
                if f.endswith((".py", ".md", ".json", ".csv", ".R", ".toml", ".cff", ".txt", ".yml", ".yaml")):
                    yield os.path.join(d, f)


@pytest.mark.parametrize("label", list(PATTERNS))
def test_no_private_identifiers_or_paths(label):
    bad = []
    for f in _files():
        if f.endswith("test_governance_cli.py"):
            continue
        with open(f, encoding="utf-8", errors="ignore") as fh:
            if PATTERNS[label].search(fh.read()):
                bad.append(os.path.relpath(f, ROOT))
    assert not bad, f"{label} found in {bad}"


def test_cli_evaluate(tmp_path):
    d = make_genes(n=1500, seed=3)
    p = tmp_path / "genes.csv"
    d.to_csv(p, index=False)
    out = tmp_path / "res.csv"
    r = subprocess.run([sys.executable, "-m", "nomibench.cli", "evaluate", "--data", str(p), "--outcome", "Y",
                        "--exposure", "E", "--covariates", "x_z,x_len,x_nd", "--linear", "x_nd", "--out", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "G1" in r.stdout and out.exists()


SPAWN = '''
import multiprocessing as mp
import sys
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)           # Windows and macOS default
    sys.path.insert(0, {tests!r})
    from conftest import make_genes
    import nomibench as nb
    from nomibench import aipw, bootstrap, negative_control
    cov = ["x_z", "x_len", "x_nd"]
    d = make_genes(n=1500, seed=3)
    a = negative_control.run(d, "Y", "E", cov, ("x_nd",), n_sets=4, seed=1, workers=2)
    assert a.equals(negative_control.run(d, "Y", "E", cov, ("x_nd",), n_sets=4, seed=1, workers=1))
    c2 = nb.calibrate(d, "Y", "E", cov, ["x_z"], linear=("x_nd",), locus="interval", locus_distance="ld",
                      n_null=3, n_power=2, power_ors=(2.0,), seed=5, workers=2)
    c1 = nb.calibrate(d, "Y", "E", cov, ["x_z"], linear=("x_nd",), locus="interval", locus_distance="ld",
                      n_null=3, n_power=2, power_ors=(2.0,), seed=5, workers=1)
    assert c1.replicates.equals(c2.replicates)
    assert aipw.d1(d, "Y", "E", cov, ("x_nd",), splits=2, entropy=1, workers=2) == \\
        aipw.d1(d, "Y", "E", cov, ("x_nd",), splits=2, entropy=1, workers=1)
    b2 = bootstrap.run(d, "Y", ["E"], cov, ("x_nd",), n_reps=2, seed=3, workers=2)
    assert b2.equals(bootstrap.run(d, "Y", ["E"], cov, ("x_nd",), n_reps=2, seed=3, workers=1))
    print("SPAWN-OK")
'''


def test_parallel_workers_under_spawn(tmp_path):
    """Worker pools must run under the spawn start method (Windows, macOS) and give the serial results."""
    script = tmp_path / "spawn_check.py"
    script.write_text(SPAWN.format(tests=os.path.join(ROOT, "tests")))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "SPAWN-OK" in r.stdout, r.stderr[-2000:]
