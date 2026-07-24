# HuberRegression -- CVXPY original (problems.py):
#     minimize  sum(huber(Xt @ beta - Y, M=1))
#
# JuMP rewrite (the standard huber epigraph, the same decomposition CVXPY's
# canonicalization uses):  huber(r, 1) = min_{u,w} u^2 + 2|w|  s.t. r = u + w,
# with |w| linearized as t >= w, t >= -w:
#
#     minimize    u'u + 2 sum(t)
#     subject to  Xt beta - y == u + w
#                 t >= w,  t >= -w

function build(d, θ)
    Xt, y = d["Xt"], vec(d["y"])
    m, n = size(Xt)
    model = Model()
    set_string_names_on_creation(model, false)
    @variable(model, beta[1:n])
    @variable(model, u[1:m])
    @variable(model, w[1:m])
    @variable(model, t[1:m])
    @constraint(model, Xt * beta .- y .== u .+ w)
    @constraint(model, t .>= w)
    @constraint(model, t .>= -w)
    @objective(model, Min, u' * u + 2 * sum(t))
    return model
end
