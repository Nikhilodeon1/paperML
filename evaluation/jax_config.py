"""JAX configuration for the identifiability analyses. Import this BEFORE any other jax import.

Two settings, both of which change results rather than only speed.

**Double precision.** The analyses here report eigenvalue spectra, condition numbers and
profile-likelihood thresholds. Single precision carries about seven significant digits, so a
condition number near 1e8 is indistinguishable from infinity and the ratio of the smallest to the
largest eigenvalue of a near-singular Fisher matrix is numerical noise. The toy tests assert an
eigenvalue ratio below 1e-8, which single precision cannot represent at all. Everything
identifiability-related therefore runs in float64.

**Persistent compilation cache.** Each worker process otherwise recompiles the jitted solve
(tens of seconds). The cache is shared across processes and across runs, which is what makes the
multi-process plan worth anything.

Note on the existing engine: `simulation/jax_engine.py` constructs its state with an explicit
`jnp.float32` dtype, so under x64 the solve upcasts on first write and JAX emits a
`FutureWarning` about an unsafe scatter cast. The values are float64 and correct; the warning is
recorded in the phase report, and the Phase 1 engine variants declare their dtype explicitly
instead of relying on promotion.
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parents[1] / ".jax_cache"


def configure(x64: bool = True, cache: bool = True) -> dict:
    """Apply the settings and return what was applied, for the provenance record."""
    import jax

    if x64:
        jax.config.update("jax_enable_x64", True)
    if cache:
        _CACHE.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(_CACHE))
        # Cache even quick compiles: the point is to avoid repeating them in every worker.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    # Reported relative to the repository root: an absolute path here would travel into every
    # result file and then into the anonymized submission package.
    return {"x64": bool(x64), "compilation_cache": _CACHE.name if cache else None,
            "jax_version": jax.__version__}


def single_thread() -> None:
    """Pin BLAS and XLA to one thread each.

    Workers are separate processes, one per core, and the user has their own work running on this
    machine. Without this, each worker spawns as many BLAS threads as there are cores and the
    machine thrashes. Must be called before numpy and jax are imported to take effect.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
