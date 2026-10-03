# H1, formulations that do not hand the tool generic known gut initial conditions.
#
# Run 1 (si_h1.jl) declared every state initial condition known, which SI.jl reads as a generic known
# value. The engine does not start from a generic value: every meal starts with an empty gut. A swap of
# k_e and k_a is an exact symmetry of the zero-initial-condition response, and a generic known gut
# initial condition breaks it, so run 1 can only be read as "identifiable if the gut starts at an
# arbitrary known state". The runs here separate the two conventions.
#
#  P1: the pre-registered formulation. Meal as an initial condition s(0)=D (known), no input, the
#      intestinal compartment g(0) unconstrained. Basal G, Id, X known.
#  P2: meal as a known input u(t); gut initial conditions unconstrained; basal G, Id, X known.
#  P3: P2 with every initial condition unknown.
using StructuralIdentifiability

odeP1 = @ODEmodel(
    G'(t) = -(1//100 + X(t)) * G(t) + (1//100) * 90 + (9//10) * ka * g(t) / 112,
    Id'(t) = -(1//10) * Id(t) + (1//20) * (G(t) - 90),
    X'(t) = -(1//20) * X(t) + (1//1000) * SI * Id(t),
    s'(t) = -ke * s(t),
    g'(t) = ke * s(t) - ka * g(t),
    y(t) = G(t)
)

odeP2 = @ODEmodel(
    G'(t) = -(1//100 + X(t)) * G(t) + (1//100) * 90 + (9//10) * ka * g(t) / 112,
    Id'(t) = -(1//10) * Id(t) + (1//20) * (G(t) - 90),
    X'(t) = -(1//20) * X(t) + (1//1000) * SI * Id(t),
    s'(t) = -ke * s(t) + u(t),
    g'(t) = ke * s(t) - ka * g(t),
    y(t) = G(t)
)

function show(res)
    for (k, v) in res; println("  ", k, " => ", v); end
    flush(stdout)
end

function by_name(ode, names)
    return [p for p in ode.parameters if string(p) in names]
end

println("=== P1: meal as initial condition s(0)=D known, no input, g(0) unconstrained ===")
known = [x for x in odeP1.x_vars if string(x) in ("G(t)", "Id(t)", "X(t)", "s(t)")]
println("known initial conditions: ", known)
println("--- assess_identifiability (global)")
show(assess_identifiability(odeP1; funcs_to_check = odeP1.parameters, known_ic = known))

println("=== P2: meal as input u(t), gut ICs unconstrained, basal G Id X known ===")
known = [x for x in odeP2.x_vars if string(x) in ("G(t)", "Id(t)", "X(t)")]
println("known initial conditions: ", known)
println("--- assess_identifiability (global)")
show(assess_identifiability(odeP2; funcs_to_check = odeP2.parameters, known_ic = known))

println("=== P3: meal as input u(t), every initial condition unknown ===")
println("--- assess_identifiability (global)")
show(assess_identifiability(odeP2; funcs_to_check = odeP2.parameters))
