# SimpleQP -- CVXPY original (problems.py):
#     minimize    0.5 * quad_form(x, P0) + q0' x
#     subject to  G0 x <= h0,  Aeq x == beq
#
# JuMP form: identical (SCS takes the quadratic objective directly).
# The dense P0 quad form is built inside @objective -- the macro layer's
# honest cost of a dense n x n quadratic.

function build(d, θ)
    P0, q0 = d["P0"], vec(d["q0"])
    G0, h0 = d["G0"], vec(d["h0"])
    Aeq, beq = d["Aeq"], vec(d["beq"])
    n = length(q0)
    model = Model()
    set_string_names_on_creation(model, false)
    @variable(model, x[1:n])
    @objective(model, Min, 0.5 * (x' * P0 * x) + q0' * x)
    @constraint(model, G0 * x .<= h0)
    @constraint(model, Aeq * x .== beq)
    return model
end
