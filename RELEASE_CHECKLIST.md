# Release checklist (v1.0.0)

These steps publish the package under your name and accounts, so only you can do them.

## 1. Fill in the metadata

| File | What to fill |
|---|---|
| `pyproject.toml` | `authors` (names, optional emails); `[project.urls]`: replace `OWNER` with your GitHub account or organisation |
| `CITATION.cff` | `authors` (in order; `orcid` optional), `repository-code`, `date-released`; `doi` once you have it (step 3) |
| `.zenodo.json` | `creators` (`"Family, Given"`, affiliation, optional `orcid`); add the paper as a `related_identifiers` entry once it has a DOI |
| `README.md` | the paper citation in the opening section and the Zenodo DOI under "Citation" |
| `LICENSE` | the copyright holder, if it should be named people or an institution rather than "the nomibench authors" |

Then check that no placeholder is left:

```bash
grep -rn "TO FILL\|OWNER\|FAMILY NAME\|GIVEN NAMES\|to be added" --exclude=RELEASE_CHECKLIST.md .
```

## 2. Create the GitHub repository

1. Create an empty repository and push this tree to it.
2. The `tests` workflow runs the fast suite on Linux, Windows and macOS with Python 3.10–3.13, the slow simulation
   checks on Linux, and the linter. All three jobs must pass before you tag the release.

## 3. Get a DOI from Zenodo

Choose one route.

**A. Reserved DOI (use this if the paper must cite the exact version).**
1. On Zenodo, start a new upload. For "Do you already have a DOI?" answer No and click "Get a DOI now!".
   The DOI is reserved but not yet registered.
2. Put the DOI into `CITATION.cff` (`doi:`), the README's "Citation" section and the manuscript's code-availability
   statement. Commit, push and tag `v1.0.0`.
3. Build the distributions (step 4), upload the sdist and wheel to the Zenodo draft, copy the metadata from
   `.zenodo.json`, and publish.

**B. GitHub integration.**
1. In Zenodo's GitHub settings, switch the repository on.
2. Create the GitHub release `v1.0.0`. Zenodo archives the tag and mints a version DOI and a concept DOI.
   It reads its metadata from `.zenodo.json`; when that file exists, Zenodo ignores `CITATION.cff`, although
   GitHub still shows it in the citation box.
3. The archived v1.0.0 cannot contain its own DOI. Add it to the README and `CITATION.cff` in a follow-up commit.

## 4. Build and check the distributions

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
python -m venv /tmp/nb && /tmp/nb/bin/pip install "dist/nomibench-1.0.0.tar.gz[test]"
tar -xzf dist/nomibench-1.0.0.tar.gz -C /tmp && cd /tmp/nomibench-1.0.0 && /tmp/nb/bin/python -m pytest -q -m "not slow"
```

Confirm that the code is the validated code: the source-tree hash in `docs/VALIDATION.md` must not change when
only metadata and documentation were edited (the command is given there).

## 5. PyPI (optional)

`twine upload dist/*` with your own PyPI API token. The version on PyPI must be the same files as on Zenodo.

## 6. Data governance

- The distributions contain no MetaBrain-derived values; `tests/test_governance_cli.py` scans for Ensembl gene
  IDs and private paths.
- `docs/VALIDATION.md` reports aggregate differences only. The reproduction script's own outputs are also
  aggregates, but they are not part of the release; do not add them, or any input table, to the repository.
