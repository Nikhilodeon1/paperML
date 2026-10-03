# Knowledge Base schema (Layer 1)

This is the "textbook" the system modules consult. It is **structured data + equations, not a trained model** (brief §2, Layer 1).

Every coefficient file is JSON with this shape:

```jsonc
{
  "system": "hepatic",              // which Layer 2 module consumes this
  "description": "...",
  "parameters": {
    "<param_name>": {
      "value": <number>,             // population point estimate (the prior mean)
      "unit": "...",
      "population_sd": <number>,      // spread across population — drives the
                                      //   prior width of confidence intervals
      "plausible_range": [lo, hi],    // hard bounds for sanity clamping
      "citation": "Author Year, ...", // REQUIRED — every value traces to a source
      "notes": "..."
    }
  }
}
```

## Rules (from the brief)

- **Every value carries a citation.** No uncited coefficients (§3).
- `population_sd` is what makes confidence intervals start wide and is the prior
  for the Layer 3 Bayesian personalization. A user's posterior narrows from here.
- Values are population-level priors. Per-user adjustments live in Layer 3, never here.
- Layer 1 only improves by manual research updates or validated cohort aggregation —
  never "passively learns" (§2.5 / §1.5 distinction).
