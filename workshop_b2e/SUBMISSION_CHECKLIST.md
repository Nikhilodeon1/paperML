# CALE-AI 2027 submission: checklist

Venue facts, re-read on 27 November 2026 from https://caleai.github.io/ and https://easychair.org/cfp/CALE-AI2027:

- **Abstract registration and paper are both due 27 November 2026. Neither page states a time zone.** No extension is
  announced.
- Full papers: 8 pages (workshop site) or 9 (EasyChair), references excluded. Use 8 as the limit.
- AAAI format. Review mode, anonymity, AI policy, and required checklists are not stated. The paper is anonymous.
- Non-archival. Permission for concurrent submission is stated for extended abstracts only; nothing is said for full papers.
- Submission: https://easychair.org/my/conference?conf=caleai2027. Contact for questions: the organizer address on the
  workshop site.
- Topic: evaluating user-facing AI interventions (causal and longitudinal human effects). **Fit risk:** this paper studies
  whether a quantity a system would show is measured well enough to be shown. It has no human outcome and no deployed
  intervention. It says so, and does not present itself as an intervention study.

## Build

```powershell
python workshop_caleai/ws_numbers.py          # final: needs the new reruns; stops if a sentence stops being true
python workshop_caleai/build.py               # writes build/Anonymous.pdf
python workshop_caleai/package.py             # writes submission/; refuses draft numbers
```

With `--draft` on the first command, the build writes `build/Draft.pdf` from the earlier runs. **Never submit Draft.pdf.**
Its trace numbers come from the solver that stopped early and its area numbers from the old protocol, while the text
describes the new protocol.

The final build needs these sweeps complete: `real`, `rand0.0/0.25/0.75` (seeds 0 and 1), `fit0.0/0.25/0.75` (seed 0),
from pod 1, and `iauc_box1`, `iauc_box2`, `iauc_box4`, from pod 2.

## Before submitting (authors)

1. Register the abstract on EasyChair. The registered text has no result numbers.
2. Read the Ethical Statement's AI paragraph and confirm that it describes what happened. It makes no claim that the
   authors verified numbers or references; add one only after doing the checks.
3. Ask the organizers, or decide yourselves, about the concurrent journal submission, and say so wherever TMLR asks.
4. Rewrite in your own words the passages listed in `author_pass.md`.
5. Check the final PDF's page count, fonts, and metadata.
