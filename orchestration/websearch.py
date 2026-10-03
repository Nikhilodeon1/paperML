"""Web search — retrieve established facts NOT covered by our modules or cited KB.

Keyless (DuckDuckGo via `ddgs`). Used by the LLM ONLY to fetch supporting facts the
simulator/knowledge-base can't provide (e.g. "recommended daily calcium", drug facts).
It never overrides a simulation number, and results are explicitly flagged web-sourced
and lower-confidence than the validated modules — consistent with the three-outcome
rule (brief §8): grounded fact retrieval, not fabrication.

Fails soft: returns an empty list on any error so a search outage never breaks a reply.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=256)
def search(query: str, max_results: int = 3) -> tuple:
    try:
        from ddgs import DDGS
        with DDGS() as d:
            hits = list(d.text(query, max_results=max_results))
    except Exception:
        return ()
    out = []
    for h in hits[:max_results]:
        out.append({"title": (h.get("title") or "")[:120],
                    "snippet": (h.get("body") or "")[:280],
                    "url": h.get("href") or h.get("url") or ""})
    return tuple(out)


def search_facts(query: str, max_results: int = 3) -> dict:
    results = list(search(query, max_results))
    return {
        "query": query,
        "results": results,
        "source": "web",
        "evidence": "weak",
        "note": ("Web-sourced supporting facts — lower confidence than the validated "
                 "simulations/knowledge base. State that these come from the web and "
                 "cite the source titles."),
    }
