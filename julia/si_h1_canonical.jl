# H1, supplementary: the same system in controllable canonical coordinates.
# With zero initial gut state the meal input reaches the rate of appearance only through
# sigma1 = ke + ka and c = ke * ka. This run asks whether (S_I, sigma1, c) are identifiable, which is
# the statement that makes the (ke, ka) swap the only remaining ambiguity: the roots of
# z^2 - sigma1 z + c are the unordered pair {ke, ka}.
using StructuralIdentifiability

ode = @ODEmodel(
    G'(t) = -(1//100 + X(t)) * G(t) + (1//100) * 90 + (9//10) * c * x1(t) / 112,
    Id'(t) = -(1//10) * Id(t) + (1//20) * (G(t) - 90),
    X'(t) = -(1//20) * X(t) + (1//1000) * SI * Id(t),
    x1'(t) = x2(t),
    x2'(t) = -c * x1(t) - sigma1 * x2(t) + u(t),
    y(t) = G(t)
)

function show(res)
    for (k, v) in res; println("  ", k, " => ", v); end
    flush(stdout)
end

println("=== C1: canonical form, gut ICs unconstrained, basal G Id X known ===")
known = [x for x in ode.x_vars if string(x) in ("G(t)", "Id(t)", "X(t)")]
println("--- assess_identifiability (global)")
show(assess_identifiability(ode; funcs_to_check = ode.parameters, known_ic = known))
println("=== C2: canonical form, every initial condition unknown ===")
println("--- assess_identifiability (global)")
show(assess_identifiability(ode; funcs_to_check = ode.parameters))
