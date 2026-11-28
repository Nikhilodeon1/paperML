# Bench2Evidence (AAAI-27 Bridge) short paper: checklist

Venue facts, read on 27 November 2026 from https://bench-to-evidence.github.io:

- Submissions due **1 December 2026, anywhere on Earth** (2 December, 3:59 am PST).
- Short paper 2 to 4 pages, or full paper 6 to 8. AAAI-27 format. **Whether references count is not stated.** The draft
  is 4 pages with references included, so it fits either reading. Keep it that way.
- Non-archival: "published work is welcome".
- Anonymity, AI-use policy, and required checklists are not stated. The paper is anonymous.
- Submission through OpenReview. The page warns that a new profile without an institutional email address can take time
  to be approved. The page's submission link is the one to use.
- AAAI-27 registration is required for every accepted submission. The Bridge is in person, 16 to 17 February 2027,
  Montreal.
- Contacts on the page: vedantk@ibm.com, pateldha@us.ibm.com.

**Fit risk:** the Bridge is about evaluating AI agents before deployment. This paper has no agent. It offers a case in
which a held-out score is taken to support a claim it does not support, and it says that transfer to agents was not
tested.

## Build

```powershell
python workshop_b2e/ws_numbers.py
```

```powershell
python workshop_b2e/build.py
```

```powershell
python workshop_b2e/package.py
```

The first command needs the new reruns and stops if a sweep is incomplete or a sentence stops being true. It needs
`real`, `rand0.0/0.25/0.75` (seeds 0 and 1), `fit0.0/0.25/0.75` (seed 0), and `iauc_box1`, `iauc_box2`, `iauc_box4`.
The third refuses draft numbers. With `--draft` on the first command the build writes `build/Draft.pdf` from the earlier
runs. **Never submit Draft.pdf:** its trace numbers come from the solver that stopped early and its area numbers from
the old protocol, while the text describes the new protocol.

The held-out prediction numbers come from the journal paper's macro files, not from these sweeps. Rebuild after those
files are updated.

## Before submitting (authors)

1. Create the OpenReview profile.
2. Confirm that the AI-use paragraph describes what happened. It makes no claim that the authors verified numbers or
   references; add one only after doing the checks.
3. Open each reference at its source. See `CITATIONS_VERIFIED.md`, which `package.py` writes.
4. Rewrite in your own words the passages listed in `author_pass.md`.
5. Say wherever TMLR asks that a non-archival short version is under review at this Bridge, if you submit both.
6. Check the final PDF's page count, fonts, and metadata.
