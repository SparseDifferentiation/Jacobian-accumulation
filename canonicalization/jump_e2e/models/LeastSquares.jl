# LeastSquares -- CVXPY original (problems.py):
#     minimize  sum_squares(A0 @ x - b0)
#
# JuMP rewrite (CVXPY's own sum_squares epigraph, written out):
#     minimize    r' r
#     subject to  r == A0 x - b0
#
# SCS accepts the quadratic objective directly (P block).  Writing
# sum((A0 x - b0).^2) instead would expand a dense n^2-term quadratic in the
# macro layer -- the residual variable is both the performant and the
# theory-identical form.

function build(d, θ)
    A0, b0 = d["A0"], vec(d["b0"])
    m, n = size(A0)
    model = Model()
    set_string_names_on_creation(model, false)
    @variable(model, x[1:n])
    @variable(model, r[1:m])
    @constraint(model, r .== A0 * x .- b0)
    @objective(model, Min, r' * r)
    return model
end
