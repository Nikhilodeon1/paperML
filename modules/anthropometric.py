"""Anthropometric module — simple, exact, cited body-metric formulas.

These are foundational, heavily-used, deterministic calculations (BMI, healthy-weight
range, body-surface-area). They are NOT simulations and carry no uncertainty — given
the inputs they are exact — so they return STRONG evidence with the source cited.

This is the pattern for "cheap coverage": a small cited formula, not a whole new
module per question. The LLM orchestrator calls these to ground answers like
"what's my BMI" instead of doing the arithmetic itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_base import load_raw
from modules.hepatic import EvidenceLevel


@dataclass
class BmiResult:
    bmi: float
    category: str
    healthy_weight_kg_range: tuple[float, float]
    evidence: EvidenceLevel
    confidence_label: str
    citations: list[str] = field(default_factory=list)
    note: str = ""


def _bmi_category(bmi: float, facts: dict) -> tuple[str, str]:
    table = facts["bmi_categories"]
    for lo, hi, label in table["value"]:
        if lo <= bmi < hi:
            return label, table["citation"]
    return "unknown", table["citation"]


def compute_bmi(weight_kg: float, height_cm: float, age: float | None = None) -> BmiResult:
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("weight_kg and height_cm must be positive")

    facts = load_raw("facts")["facts"]
    h_m = height_cm / 100.0
    bmi = weight_kg / (h_m * h_m)
    category, cat_cite = _bmi_category(bmi, facts)

    lo_bmi, hi_bmi = facts["healthy_bmi_range"]["value"]
    healthy_range = (round(lo_bmi * h_m * h_m, 1), round(hi_bmi * h_m * h_m, 1))

    # BMI classification is not valid for children/adolescents.
    if age is not None and age < 18:
        evidence = EvidenceLevel.WEAK
        note = ("BMI value is exact, but the adult category labels do not apply under "
                "18y — pediatric BMI-for-age percentiles should be used instead.")
    else:
        evidence = EvidenceLevel.STRONG
        note = ""

    return BmiResult(
        bmi=round(bmi, 1),
        category=category,
        healthy_weight_kg_range=healthy_range,
        evidence=evidence,
        confidence_label="exact formula from your height and weight",
        citations=[cat_cite, facts["healthy_bmi_range"]["citation"]],
        note=note,
    )
