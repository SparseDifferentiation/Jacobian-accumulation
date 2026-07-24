# SimpleLP -- CVXPY original (problems.py):
#     minimize    c @ x
#     subject to  0 <= x,  x <= 1
#
# JuMP form: identical; the box goes on as variable bounds, JuMP's idiomatic
# form (MOI bridges them to SCS's nonnegative cone rows, the same rows CVXPY
# emits as constraints).
#
# Performance tips applied: macro-built expressions, string names off.

function build(d, θ)
    c = vec(d["c"])
    n = length(c)
    model = Model()
    set_string_names_on_creation(model, false)
    @variable(model, 0 <= x[1:n] <= 1)
    @objective(model, Min, c' * x)
    return model
end
