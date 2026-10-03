"""Cross-system dependency graph (brief 2.5).

Modules do not talk to each other automatically — connections are DECLARED here, by
us, with real evidence and a Layer-1-backed coefficient behind every edge. The
orchestration LLM (Layer 4) will consult this graph to chain module calls in the
right order; it never invents an edge at runtime.

Hard guardrail (brief 2.5 / 8): a cross-system answer is only produced if a declared
edge with real coefficients supports it. A fluent-sounding connection with no edge is
the most dangerous failure mode in the system — we refuse it instead.
"""
