"""The authored dependency graph + the chained query runner.

An `Edge` declares that one module's output feeds another module's input, modified by
a documented, graded, cited relationship. The graph is static and team-authored. The
runner walks declared edges only; `has_edge` is the guardrail Layer 4 must consult
before attempting any cross-system reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_base import load_raw, load_system, parameter_from_dict
from modules.hepatic import Drink, EvidenceLevel, compute_bac
from modules.sleep import SleepModel, SleepPrediction
from modules.stress import StressModel


@dataclass(frozen=True)
class Edge:
    """A declared cross-system relationship."""

    name: str
    source: str            # source module name
    target: str            # target module name
    target_input: str      # which target input the source feeds
    evidence: EvidenceLevel
    citation: str
    notes: str = ""


# --- The authored graph -----------------------------------------------------
# ONLY edges with real evidence (brief 2.5). Each is graded; weak/mixed-evidence
# edges (e.g. blood pressure -> concentration) are deliberately NOT declared in v1.
EDGES: tuple[Edge, ...] = (
    Edge(
        name="alcohol_to_sleep",
        source="hepatic",
        target="sleep",
        target_input="alcohol_gkg_bedtime",
        evidence=EvidenceLevel.STRONG,
        citation="Ebrahim et al. 2013, Alcohol Clin Exp Res",
        notes="Residual alcohol burden at bedtime suppresses REM and fragments sleep.",
    ),
    Edge(
        name="stress_to_sleep",
        source="stress",
        target="sleep",
        target_input="sleep_efficiency",
        evidence=EvidenceLevel.WEAK,   # graded moderate: direction solid, dose-response less so
        citation="Kalmbach et al. 2018, J Sleep Res; MMASH check (evaluation/calibrate_mmash.py)",
        notes="Pre-sleep arousal/stress reduces sleep continuity. Applied as a post-hoc "
              "efficiency modifier. WEAK grade corroborated on real data: MMASH (n=21) "
              "shows weak, directionally-consistent stress->sleep associations — enough "
              "to keep the edge, not to upgrade it.",
    ),
)


def has_edge(source: str, target: str) -> bool:
    """Guardrail: is there a declared edge connecting these systems?"""
    return any(e.source == source and e.target == target for e in EDGES)


def get_edge(source: str, target: str) -> Edge:
    for e in EDGES:
        if e.source == source and e.target == target:
            return e
    raise ValueError(
        f"No declared edge {source!r} -> {target!r}. Refusing to connect these "
        f"systems: a cross-system claim without an authored, evidence-backed edge "
        f"is not allowed (brief 2.5)."
    )


def _bac_to_gkg(bac: float, sex: str) -> float:
    """Convert blood alcohol concentration (g/100mL) to alcohol burden per kg of body
    mass, using the Widmark distribution ratio from Layer 1:

        central_g = BAC * (r * W * 10)  =>  central_g / W = BAC * r * 10  [g/kg]
    """
    r = load_system("hepatic")["widmark_r_male" if sex == "male" else "widmark_r_female"].value
    return max(0.0, bac * r * 10.0)


@dataclass
class ChainResult:
    bedtime_bac: float
    alcohol_gkg_bedtime: float
    sleep: SleepPrediction
    edge: Edge
    evidence: EvidenceLevel
    confidence_label: str
    citations: list[str]


def run_alcohol_then_sleep(
    drinks: list[Drink],
    weight_kg: float,
    sex: str,
    bedtime_hour: float,
    age: float,
    sleep_model: SleepModel | None = None,
    **sleep_kwargs,
) -> ChainResult:
    """Chained query: hepatic -> (declared edge) -> sleep.

    1. Hepatic computes the BAC curve.
    2. The declared edge converts BAC at bedtime into the sleep module's
       `alcohol_gkg_bedtime` input (Layer-1-backed conversion).
    3. Sleep predicts the night's architecture under that alcohol burden.

    The overall evidence is the weakest link across the chain (brief 8), and the
    label carries the sleep model's provisional-synthetic caveat through.
    """
    edge = get_edge("hepatic", "sleep")  # raises if undeclared

    bac = compute_bac(drinks, weight_kg=weight_kg, sex=sex,
                      horizon_h=max(bedtime_hour + 1, 12.0))
    idx = int(min(range(len(bac.times_h)),
                  key=lambda i: abs(bac.times_h[i] - bedtime_hour)))
    bedtime_bac = float(bac.bac_median[idx])
    gkg = _bac_to_gkg(bedtime_bac, sex)

    model = sleep_model or SleepModel().fit()
    sleep = model.predict(age=age, alcohol_gkg_bedtime=gkg, **sleep_kwargs)

    # Weakest link across hepatic (strong), the edge (strong), and the sleep model.
    levels = [bac.evidence, edge.evidence, sleep.evidence]
    overall = (EvidenceLevel.NONE if EvidenceLevel.NONE in levels
               else EvidenceLevel.WEAK if EvidenceLevel.WEAK in levels
               else EvidenceLevel.STRONG)

    citations = list(dict.fromkeys(bac.citations + [edge.citation] + sleep.citations))
    label = (f"chain hepatic->sleep ({edge.evidence.value} edge); "
             f"{sleep.confidence_label}")

    return ChainResult(
        bedtime_bac=bedtime_bac,
        alcohol_gkg_bedtime=gkg,
        sleep=sleep,
        edge=edge,
        evidence=overall,
        confidence_label=label,
        citations=citations,
    )


@dataclass
class StressSleepResult:
    stress_index: float
    baseline_efficiency: float
    adjusted_efficiency: float
    sleep: SleepPrediction
    edge: Edge
    evidence: EvidenceLevel
    confidence_label: str
    citations: list[str]


def _stress_to_sleep_coef() -> float:
    mod = load_raw("stress")["cross_system_modifiers"]["stress_to_sleep"]
    return parameter_from_dict("eff_per_stress", mod["efficiency_per_stress_point"]).value


def run_stress_then_sleep(
    *,
    heart_rate: float,
    rmssd: float,
    eda: float,
    age: float,
    sleep_model: SleepModel | None = None,
    stress_model: StressModel | None = None,
    **sleep_kwargs,
) -> StressSleepResult:
    """Chained query: stress -> (declared edge) -> sleep.

    The stress module scores bedtime arousal from wearable signals; the declared edge
    applies a Layer-1 efficiency modifier to the sleep prediction. Graded WEAK (brief
    2.5): the direction is well established but the dose-response is not, so the label
    flags low confidence.
    """
    edge = get_edge("stress", "sleep")  # raises if undeclared

    sm = stress_model or StressModel().fit()
    stress = sm.predict(heart_rate=heart_rate, rmssd=rmssd, eda=eda)

    slm = sleep_model or SleepModel().fit()
    sleep = slm.predict(age=age, **sleep_kwargs)

    baseline_eff = sleep.metrics["sleep_efficiency"]
    delta = _stress_to_sleep_coef() * stress.stress_index   # negative
    adjusted_eff = float(min(100.0, max(0.0, baseline_eff + delta)))

    citations = list(dict.fromkeys([edge.citation] + stress.citations + sleep.citations))
    label = (f"chain stress->sleep ({edge.evidence.value} edge, low confidence); "
             f"{sleep.confidence_label}")

    return StressSleepResult(
        stress_index=stress.stress_index,
        baseline_efficiency=baseline_eff,
        adjusted_efficiency=adjusted_eff,
        sleep=sleep,
        edge=edge,
        evidence=EvidenceLevel.WEAK,   # weakest link: the edge is graded moderate
        confidence_label=label,
        citations=citations,
    )
