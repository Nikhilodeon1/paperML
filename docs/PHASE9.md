# Phase 9 (Amendment 5, TMLR revision): what to run and what happens after

## Order of operations

1. Commit `PREREG_AMENDMENT_5.md` **alone and first**, then commit and push everything else. Result folders are
   named by the commit, so nothing may be committed or pulled while the queue runs.
2. On the node: `git pull`, check `results/A4_fisher` exists (the replica needs the stored CGMacros fits; if the
   node's `results/` is empty, upload `a4_fisher.tgz` made locally with `tar czf a4_fisher.tgz results/A4_fisher`).
3. Run `scripts/pod_phase9.sh` (about six hours on 30 workers; resumable; stop it whenever, then
   `bash scripts/pod_phase9.sh archive`).
4. Download `phase9_results.tgz`, unpack into the local `paperML/`, copy without overwriting
   (`cp -rn`), then:
   `python -m evaluation.make_paper_assets --latex` (summaries, freeze, macros, figures, tables, lint, PDF).

## Priority inside the queue

H10 sweeps, polished coordinate profiles (H21, H11), coordinate replica (H18), synthetic study (H19, H6),
Shanghai cross-validation (H20), second model class (H22), replica seeds 1 to 4 (H17), report-only runs.
Anything not finished at the second freeze is dropped and the paper says it was not run.

## After the results

* Replace every `\todo{}` in `paper/sec_results.tex` and `paper/app_register.tex` using the verdicts.
* Check the claims table in the introduction against the verdicts (H20 and H22 in particular).
* `python -m evaluation.tmlr_build` lists remaining placeholders and citations to verify.

## Rebuilding the environment on a new node (the venv lives in /tmp and does not survive)

```bash
cd ~/paperML
python3 --version                      # needs 3.11 or newer
python3 -m venv /tmp/venv && . /tmp/venv/bin/activate && pip install -U pip
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
grep -v '^torch==' requirements-aistats.txt > /tmp/req.txt && pip install -r /tmp/req.txt
python -c "import jax, equinox, diffrax; print('env ok')"
```
