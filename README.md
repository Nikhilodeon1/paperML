# Horizon — ML Core

Personal physiological simulation engine. See `horizon_ml_architecture_brief.md` for the full design.

## Quickest start: the ONE interface to test the model

```bash
.venv\Scripts\activate
uvicorn api.app:app --reload --port 8000
```
Open **http://127.0.0.1:8000/** — a single test page with your profile, a question
box, example queries (including the cross-system "2 shots of vodka → sleep" case),
and a rendered answer with **plain-language explanation, the exact indicators used,
and citations**. This is what a real app's backend would expose; everything (Flutter
app, curl, this page) talks to the same `/ask` endpoint.

**Default computation is 100% ML/mechanistic — no LLM in the loop.** Layers 1–3 (the
actual physiology) are always equations + gradient-boosted trees + Bayesian updating;
there is no LLM anywhere in that path. The only optional LLM is in Layer 4 *routing*
(turning free-text into a tool call), and it's off by default — check "Route via
Gemini" only if you want to test that separately. See "How accurate is this?" below.

## Architecture (four layers)

1. **`knowledge_base/`** — static, shared physiological constants/equations with citations. Not a model. JSON + small loader.
2. **`modules/`** — one mechanistic/statistical model per physiological system. Pulls coefficients from Layer 1.
3. **`personalization/`** — Bayesian parameter updating (per-user posteriors). *Not yet built.*
4. **`orchestration/`** — function-calling LLM interface. *Not yet built — per Section 6, do not touch until a module is validated.*

## Current status

Following Section 6 of the brief: each module is fully backtested before the next.

- ✅ Layer 1 schema + hepatic & metabolic coefficients (all cited)
- ✅ **Hepatic module** (BAC pharmacokinetics) — validated
  - Backtest vs analytic Widmark: peak BAC mean err **0.64%** (max 1.55%), time-to-sober mean **2.29%** (max 5.19%)
  - Calibration: mean coverage error **1.29%** across 50–95% intervals
- ✅ **Metabolic module** (Mifflin-St Jeor + energy balance) — validated
  - RMR exact vs hand-computed; maintenance drift 0.03 kg/yr; adaptation below naive linear; long-horizon (>180 d) auto-downgraded to weak evidence
  - Calibration: mean coverage error **0.63%** across 50–95% intervals
- ✅ **Layer 3 Bayesian personalization** (metabolic) — recursive Gaussian / Kalman updating
  - Recovers a hidden RMR multiplier from weigh-ins; posterior + projection CI tighten automatically with more data (see `demo_personalization`)
  - Posterior SD feeds the module directly, replacing the `n_observations` stand-in
- ✅ **Sleep module** (ML) — **real age-baseline on PhysioNet Sleep-EDF + cited behavioural modifiers** (no synthetic in deployed model)
  - Quantile GBMs fit on 153 real nights; held-out MAE beats baseline by 17% avg; PI coverage ~84–90%
  - Behavioural effects (alcohol, caffeine, exercise, screen, regularity) applied from cited research (Layer 1)
  - Alcohol→sleep edge directionally correct; evidence now **strong** (real data), not provisional
- ✅ **Cardiovascular module** (Framingham General CVD, D'Agostino 2008) — mechanistic, no dataset
  - 10-yr CVD risk + CI + heart-age; healthy 40 y M = 2.9% (textbook), high-risk 70 y = >60%
  - All monotonicity / archetype / out-of-range-evidence checks pass
- ✅ **Cross-system dependency graph** (`pipeline/`) — alcohol→sleep edge, declared & graded
  - Chains hepatic→sleep; guardrail refuses any undeclared edge (no fabricated connections)
  - Demo: `evaluation/demo_chain.py` (6 drinks → REM 22%→15%, awakenings 1.8→2.9)
- ✅ **Stress module** (ML) — **supervised classifier trained on WESAD** (labelled wrist-E4). `stress_index = 100 × P(stress)`
  - Leave-one-subject-out: **87.9% balanced accuracy, ROC-AUC 0.983** (chance 50%); HR/EDA the reliable drivers
  - Falls back to the HR-elevation composite only if WESAD is absent; evidence now **strong**
- ✅ **Layer 3 hepatic personalization** — personal elimination rate via the same Kalman updater
- ✅ **Multi-backend LLM orchestration** — **Groq (gpt-oss)** + **Gemini** + deterministic, unified via `orchestration/orchestrate.route_llm` (priority: Groq → Gemini → rules; numbers grounded identically in all)
  - Routes NL questions to modules; chains cross-system; returns honest "no model" third outcome
  - `llm_gemini.route_with_gemini` does LLM routing only (1 call/question), dispatch + grounded explanation stay local; falls back to rules without a key
  - Privacy: the LLM only sees the question + already-computed numbers, never raw health logs
  - Demos: `evaluation/demo_orchestration.py` (rules), `evaluation/demo_gemini.py` (LLM)
- ✅ **Real-data ingestion scaffold** — `data/` + swappable `*_data_real.py` loaders (drop-in for `Model(data_fn=...)`)
- ✅ **Real data wired in** (datasets live in `../data/`, i.e. `DigiTwin/data/`)
  - **Sleep-EDF** (153 nights): architecture norms confirmed; **age→deep-sleep slope −0.175 pp/yr matches KB −0.180** → KB cohort-refined. `evaluation/calibrate_sleep_edf.py`
  - **NHANES 2017-18** (~9k people): Framingham risk monotonic by age (2.6→19%), mean RMR 1530 kcal, intake/TDEE 0.98 — all plausible. `evaluation/calibrate_nhanes.py`
  - **Wearable Exam Stress** (597 windows, Empatica E4): HR elevated under exam stress (94.7 vs 68 resting), HR↔RMSSD arousal signature −0.24 (after HRV artifact rejection). `evaluation/calibrate_exam_stress.py`
  - Real loaders: `sleep_data_real.load_sleep_edf`, `stress_data_real.load_exam_features` (working)
- ✅ **9/9 backtests + calibrations pass** (synthetic and real data); **78 unit tests passing**
- ✅ **Gemini key config** — drop your key in `ml/.env`; live routing verified (parses "5 pints" → 6.7 drinks)
- ✅ **stress→sleep edge** — declared, graded WEAK, wired into `pipeline/graph.py` (`run_stress_then_sleep`)
- ✅ **ML core complete**: 5 modules, personalization, 2 cross-system edges, LLM + rule routing, all validated on real data
- ✅ **FastAPI backend + test console** (`api/`) — the one interface described above; deterministic router is the default, Gemini is opt-in per request
- ✅ **Explainability layer** — every answer carries plain-language `explanation` bullets (derived from the module's own ranked drivers, never invented) and an `indicators` dict of the exact numbers used
- ✅ **Aggregate accuracy report** (`evaluation/full_report.py`) — one command rolls up all 9 backtests/calibrations (synthetic + real data) into a single scoreboard
- ✅ **95 unit tests passing**
- ⬜ Improve models (the "begin improving" phase)

## How accurate is this? (evaluate it yourself)

```bash
python -m evaluation.full_report
```
One command, one scoreboard: runs all 9 backtests/calibrations (mechanistic accuracy,
ML calibration, and 3 checks against REAL downloaded data — NHANES ~9k people,
Sleep-EDF 153 nights, wearable exam-stress 597 windows) and prints **N/N passed**
with each one's own headline metric. There's no single "accuracy %" because a
mechanistic module (relative error vs a known equation) and an ML module (interval
calibration) aren't measured the same way — the report shows each in its own units
rather than faking a composite score. Currently: **9/9 pass**. Headlines:
- Hepatic BAC: 0.64% mean error vs the analytic Widmark equation
- Both calibration harnesses: <1.3% mean gap between claimed and actual confidence
- NHANES: Framingham risk rises monotonically 2.6%→19% across real age deciles
- Sleep-EDF: our literature-derived age→deep-sleep coefficient (−0.18) matches the
  real cohort fit (−0.175) almost exactly

### Run everything

```bash
python -m evaluation.full_report             # <- the aggregate scoreboard, start here
python -m evaluation.backtest_hepatic
python -m evaluation.calibrate_hepatic
python -m evaluation.backtest_metabolic
python -m evaluation.calibrate_metabolic
python -m evaluation.backtest_cardiovascular
python -m evaluation.demo_personalization
python -m evaluation.demo_chain
python -m evaluation.demo_orchestration
python -m evaluation.calibrate_sleep_edf     # real Sleep-EDF validation
python -m evaluation.calibrate_nhanes        # real NHANES validation
python -m evaluation.calibrate_exam_stress   # real wearable-stress validation
python -m pytest tests/ -q
```

## Test everything (playground)

```bash
python -m evaluation.playground              # capabilities tour — all 10 features, offline
python -m evaluation.playground --chat       # type your own questions (deterministic, free)
python -m evaluation.playground --chat --gemini   # type questions, routed via Gemini (uses API)
```

## Using the Gemini LLM router

```bash
cp .env.example .env        # then edit .env and paste your key
python -m evaluation.demo_gemini
```
Get a key at https://aistudio.google.com/apikey. Without a key the router falls back
to deterministic keyword routing, so everything still runs.

## Layers built so far

```
knowledge_base/   Layer 1  hepatic, metabolic, cardiovascular, sleep, stress JSON (+ loader)
modules/          Layer 2  hepatic, metabolic, cardiovascular (mechanistic);
                           sleep, stress (ML, synthetic — provisional) + *_data_real loaders
personalization/  Layer 3  gaussian_bayes (generic Kalman); metabolic + hepatic posteriors
pipeline/         graph    cross-system edges + chained query runner (alcohol→sleep, stress→sleep)
orchestration/    Layer 4  tool schemas, deterministic router (default), Gemini router (opt-in),
                           explainability layer (plain-language bullets + indicators), .env config
api/                       FastAPI backend + single-page test console (the "one interface")
data/                      downloaded datasets go here (gitignored; see data/README.md)
evaluation/                backtests, calibration harnesses, demos, full_report.py (accuracy scoreboard)
tests/                     95 tests
```

## Layout

```
ml/
  knowledge_base/      Layer 1: constants, equations, citations
  modules/             Layer 2: system modules (hepatic first)
  evaluation/          backtesting + calibration harness (build before UI)
  tests/               unit tests
  requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m evaluation.backtest_hepatic
```
