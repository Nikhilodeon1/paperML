# H1: structural identifiability of (S_I, k_e, k_a) from the full glucose trace.
# Model: Bergman minimal model plus the two-compartment gut cascade, in the active regime
# (glucose above basal), written as polynomial ODEs for StructuralIdentifiability.jl.
# Meal rate u(t) is a known input. Initial conditions are treated as known (basal, empty gut),
# which is the engine setting: every meal starts from the basal fixed point.
using StructuralIdentifiability

# Variant A: only S_I, ke, ka unknown; physiological constants fixed at representative values.
odeA = @ODEmodel(
    G'(t) = -(1//100 + X(t)) * G(t) + (1//100) * 90 + (9//10) * ka * g(t) / 112,
    Id'(t) = -(1//10) * Id(t) + (1//20) * (G(t) - 90),
    X'(t) = -(1//20) * X(t) + (1//1000) * SI * Id(t),
    s'(t) = -ke * s(t) + u(t),
    g'(t) = ke * s(t) - ka * g(t),
    y(t) = G(t)
)

# Variant B: the six physiological constants are also unknown, to show the result is not an artifact
# of the representative values. S_I then enters only through the product with the other gains.
odeB = @ODEmodel(
    G'(t) = -(Sg + X(t)) * G(t) + Sg * 90 + (9//10) * ka * g(t) / Vg,
    Id'(t) = -n * Id(t) + gam * (G(t) - 90),
    X'(t) = -p2 * X(t) + p3 * SI * Id(t),
    s'(t) = -ke * s(t) + u(t),
    g'(t) = ke * s(t) - ka * g(t),
    y(t) = G(t)
)

function report(name, ode)
    println("=== ", name, " ===")
    println("parameters: ", ode.parameters)
    println("states: ", ode.x_vars)
    funcs = ode.parameters
    println("--- assess_local_identifiability (initial conditions generic unknown: conservative)")
    res = assess_local_identifiability(ode; funcs_to_check = funcs)
    for (k, v) in res; println("  ", k, " => ", v); end
    println("--- assess_identifiability, global (all initial conditions known)")
    res = assess_identifiability(ode; funcs_to_check = funcs, known_ic = ode.x_vars)
    for (k, v) in res; println("  ", k, " => ", v); end
    flush(stdout)
end

report("A: S_I, ke, ka unknown, constants known", odeA)
report("B: constants also unknown", odeB)
