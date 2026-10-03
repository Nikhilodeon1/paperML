# Structural identifiability runs (H1)

Julia 1.13.1, StructuralIdentifiability.jl 0.5.34. Model: Bergman minimal model, two-compartment gut
cascade, active regime, output y = glucose. Raw output for every run is committed next to the script.

| run | script | formulation | known initial conditions | S_I | k_e, k_a |
| --- | --- | --- | --- | --- | --- |
| 1 | `si_h1.jl` (A) | meal as input u(t); constants fixed | all state ICs known, generic | global | global |
| 1B | `si_h1.jl` (B) | as 1, six constants also unknown | all state ICs known, generic | non-identifiable | global |
| P1 | `si_h1_gut_ic.jl` | **planned**: meal as `s(0)=D`, no input | G, Id, X, s known; g(0) generic | global | **global** |
| P2 | `si_h1_gut_ic.jl` | meal as input u(t) | G, Id, X known; gut ICs free | global | **local** |
| P3 | `si_h1_gut_ic.jl` | meal as input u(t) | none known | global | **local** |
| C1 | `si_h1_canonical.jl` | canonical coordinates (sigma1, c) | G, Id, X known | global | sigma1, c global |
| C2 | `si_h1_canonical.jl` | canonical coordinates | none known | global | sigma1, c global |

Raw outputs: `si_h1_run1_output.txt`, `si_h1_gut_ic_output.txt`, `si_h1_canonical_output.txt`.
The local assessment in run 1 treats initial conditions as generic unknown (the tool does not accept known
initial conditions for local assessment): all three parameters locally identifiable.

Reading: the tool gives "global" for k_e and k_a exactly when the intestinal initial condition is a
generic known value (1, P1). The engine starts with an empty gut, for which the exchange is an exact
symmetry (`evaluation/swap_check.py`, `evaluation/gut_sweep.py`).
