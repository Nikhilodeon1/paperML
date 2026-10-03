"""Layer 2 — System modules.

One calibrated mechanistic/statistical model per physiological system. Each pulls
its baseline coefficients from Layer 1 (knowledge_base) and exposes a typed result
the orchestration layer (Layer 4) can consume via function calling.

Per brief section 6: build and validate ONE module fully before adding a second.
Hepatic (BAC pharmacokinetics) is first.
"""
