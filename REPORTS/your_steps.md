# Your steps, in order

Follow top to bottom. Steps 1 to 4 start the reruns. Nothing after step 6 can start until the results are back.

## 1. Push the branch (on your computer)

```powershell
git -C C:\Users\nikhi\Codes\RES\src\DigiTwin\paperML push -u origin tmlr-revision
```

## 2. Start three Colab CPU pods (8 cores each)

On each pod run this first and approve Drive:

```python
from google.colab import drive; drive.mount('/content/drive')
```

Then run this cell. Change only `P`: use 1 on the first pod, 2 on the second, 3 on the third.

```bash
%%bash
P=1
cd /content
[ -d paperML ] || git clone -q https://github.com/Nikhilodeon1/paperML.git
cd paperML
pkill -f pod_phase21; pkill -f gapfill; pkill -f stage2_queue; pkill -f evaluation.runner; pkill -f multiprocessing; sleep 5
git fetch -q origin tmlr-revision && git checkout -q -f FETCH_HEAD && git log -1 --oneline
pip install -q -r requirements-aistats.txt || pip install -q "jax[cpu]==0.11.0" diffrax equinox optax scipy scikit-learn numpy pandas
tar xzf a4_fisher.tgz
python -m evaluation.cohort_cache verify
mkdir -p logs
if [ -f /content/drive/MyDrive/stage2_pod${P}_results.tgz ]; then echo "restoring saved results"; tar xzf /content/drive/MyDrive/stage2_pod${P}_results.tgz --wildcards 'results/*'; fi
nohup python -m evaluation.stage2_queue --pod $P --workers $(nproc --all) > logs/stage2.out 2>&1 &
sleep 40; tail -n 5 logs/stage2.out
```

**Good output:** the commit shown matches the one in my last message, three cohort checks say `ok`, and a line starts with
`=== ... [1/`. Anything else: paste it to me.

## 3. Check progress (any time)

```bash
!grep '^===' /content/paperML/logs/stage2.out | tail -n 2; tail -n 1 /content/paperML/logs/stage2.out
```

## 4. If a pod disconnects

Mount Drive and run the step 2 cell again with the same `P`. Saved runs are restored and only missing work is redone.

## 5. First check-in (after about 4 to 7 hours)

Download the three `stage2_pod*_results.tgz` files from Drive (they update after every finished run), put them in
`paperML\`, and tell me. **Leave the pods running.** I check completeness and apply the rules to whatever is complete.

| Pod | Finishes first | Rough time |
|---|---|---|
| 2 | the three main iAUC runs (the abstract's first claim) | 2 h |
| 1 | the twelve trace runs for Section 5.4 | 4 h |
| 1 | then calibration and Dalla Man runs | 7 h |

All times are rough estimates and could be off by a factor of two.

## 6. Final download (about 12 to 16 hours after the start)

Each pod prints `STAGE2 POD n DONE; unfinished: none`. If it lists unfinished runs, run the step 2 cell again. Then
download the three files again, replace the old ones in `paperML\`, tell me, and shut the pods down.

## 7. What I then do (nothing for you)

Merge, check completeness, apply every rule, point the paper's numbers at the new runs, redraw figures, rebuild, run
the linter, the number check and the full test suite, and tag one commit. I report each rule's outcome to you.

## 8. Your decisions and checks (after step 7)

1. **Send the reviewer** the new PDF with the rule outcomes.
2. **Write by hand,** from `REPORTS/claim_ledger.md`: abstract, introduction, related work, Section 5.4, discussion,
   conclusion. Start from a blank page, not from the current draft.
3. **Check references** at their primary sources. Cited with only their existence checked: bellman1970structural,
   rothenberg1971identification, venzon1988method, efron1987better, holm1979simple, wilson1927probable,
   byrd1995limited, white1982maximum, dallaman2002oral, brazeau2013carbohydrate, cook1994simulation,
   albers2017personalized.
4. **AI-use statement:** make it describe what happened. Add the sentence that the authors verified numbers and
   references only after you have done both.
5. **One outside reader** for the abstract and Section 5.
6. **Anonymity:** tell me the names and strings to forbid so I can build and check the anonymous package. It must leave
   out git history, `*.failed.json`, `results_return/logs`, and `workshop_caleai/`, some of which contain your local user
   path.
7. **Submit** once there are no `[pending]` cells, no undefined references, and the checks in step 7 pass.

## CALE-AI (abstract and paper both due 27 November 2026; no time zone stated)

1. **Register the abstract on EasyChair now** (title and abstract without result numbers).
2. When pod 1 has finished its first 12 runs and pod 2 its first 3, download `stage2_pod1_results.tgz` and
   `stage2_pod2_results.tgz` into `paperML` and tell me. I merge them and run:

   ```powershell
   python workshop_caleai/ws_numbers.py
   ```

   ```powershell
   python workshop_caleai/build.py
   ```

   ```powershell
   python workshop_caleai/package.py
   ```

   The first command stops if a sweep is incomplete or a sentence of the paper stops being true.
3. Confirm the AI-use paragraph, rewrite the passages in `workshop_caleai/author_pass.md`, and submit
   `workshop_caleai/submission/Anonymous.pdf`. **Never submit `Draft.pdf`.**

Details and risks: `workshop_caleai/SUBMISSION_CHECKLIST.md`.
