# SVMWithL1Regularization -- CVXPY original (problems.py):
#     minimize  sum(pos(1 - Y .* (X @ beta - v))) / m + lambda * norm(beta, 1)
#
# JuMP rewrite (hinge + l1 epigraphs, as in CVXPY's canonicalization):
#     minimize    sum(h)/m + lambda * sum(s)
#     subject to  h >= 1 - y .* (X beta - v),  h >= 0
#                 s >= beta,  s >= -beta
#
# theta = (lambda,) baked in at build time (no parameter objects in JuMP).

function build(d, θ)
    X, y = d["X"], vec(d["y"])
    lambda = θ[1] isa AbstractArray ? first(θ[1]) : θ[1]
    m, n = size(X)
    model = Model()
    set_string_names_on_creation(model, false)
    @variable(model, beta[1:n])
    @variable(model, v)
    @variable(model, h[1:m] >= 0)
    @variable(model, s[1:n])
    @constraint(model, h .>= 1 .- y .* (X * beta .- v))
    @constraint(model, s .>= beta)
    @constraint(model, s .>= -beta)
    @objective(model, Min, sum(h) / m + lambda * sum(s))
    return model
end
