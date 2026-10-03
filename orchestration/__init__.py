"""Layer 4 — Orchestration.

The ONLY layer where an LLM belongs, and its job is narrow (brief 2, Layer 4):
parse the question, decide which module(s) to call with what parameters, call them
(function/tool-calling), and translate the structured numeric output back into
natural language. The LLM never generates a number — it routes and explains numbers
the modules already computed.

This package is split so the LLM stays thin and replaceable:
  - `tools.py`  : function/tool schemas + a deterministic dispatcher that actually
                  calls Layer 2/Layer 3/pipeline. This is testable WITHOUT any LLM.
  - `router.py` : a deterministic, rule-based router (keyword intent) used as the
                  reference implementation and offline fallback. An LLM can replace
                  the routing step later, but must call the SAME tools.

Per brief 6, this layer is built only now that multiple modules are validated and the
dependency graph exists. The LLM call itself is intentionally NOT wired here — see
`router.py` for where it plugs in.
"""
