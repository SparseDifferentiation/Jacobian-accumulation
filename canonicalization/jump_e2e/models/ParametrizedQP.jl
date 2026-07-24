# ParametrizedQP -- CVXPY original (problems.py):
#     A = Parameter((m, n)); b = Parameter(m)
#     minimize    sum_squares(A @ x - b)
#     subject to  0 <= x <= 1
#
# JuMP rewrite: r = A x - b residual (sum_squares epigraph), box as variable
# bounds.  JuMP has no parameter objects, so theta = (A, b) is baked into the
# model at build time -- a fresh build per parameter draw is JuMP's honest
# one-shot path (see ParametricOptInterface.jl for the ecosystem's answer).

function build(d, θ)
    A, b = θ[1], vec(θ[2])
    m, n = size(A)
    model = Model()
    set_string_names_on_creation(model, false)
    @variable(model, 0 <= x[1:n] <= 1)
    @variable(model, r[1:m])
    @constraint(model, r .== A * x .- b)
    @objective(model, Min, r' * r)
    return model
end
