# Lowered problem models for the JuMP/MOI comparison.
#
# Same z layout and conic row order as the CasADi Specs and _lowered_data.py.
# Constraints are written as vector-affine-in-cone (b - A z in Zeros/Nonnegatives)
# in CVXPY's row order, which maps 1:1 onto Clarabel's native MOI form -- no
# bridging, so what copy_to assembles IS the (P, q, A, b) we verify against.
#
# JuMP never differentiates anything here: affine/quadratic coefficients flow
# from the user's arrays into MOI's MatrixOfConstraints-style storage during
# copy_to.  Structure is a priori, as in SparseDiffEngine -- the comparison
# measures pure data-structure assembly, the "no-AD" end of the spectrum.

_jscalar(d, k) = (v = d[k]; v isa AbstractArray ? first(v) : v)

function jump_simple_qp(d)
    P0, q0 = d["P0"], vec(d["q0"])
    G0, h0 = d["G0"], vec(d["h0"])
    Aeq, beq = d["Aeq"], vec(d["beq"])
    n = length(q0)
    model = JuMP.Model()
    JuMP.@variable(model, z[1:n])
    JuMP.@objective(model, Min, 0.5 * z' * P0 * z + q0' * z)
    JuMP.@constraint(model, beq - Aeq * z in MOI.Zeros(length(beq)))
    JuMP.@constraint(model, h0 - G0 * z in MOI.Nonnegatives(length(h0)))
    return model
end

function jump_least_squares(d)
    A0, b0 = d["A0"], vec(d["b0"])
    m, n = size(A0)
    model = JuMP.Model()
    JuMP.@variable(model, t[1:m])          # z = [t; x]: t declared first
    JuMP.@variable(model, x[1:n])
    JuMP.@objective(model, Min, t' * t)
    JuMP.@constraint(model, b0 + t - A0 * x in MOI.Zeros(m))
    return model
end

function jump_parametrized_qp(d, Ap, bp)
    m, n = size(Ap)
    model = JuMP.Model()
    JuMP.@variable(model, t[1:m])          # z = [t; x]
    JuMP.@variable(model, x[1:n])
    JuMP.@objective(model, Min, t' * t)
    JuMP.@constraint(model, bp + t - Ap * x in MOI.Zeros(m))
    JuMP.@constraint(model, vcat(x, 1 .- x) in MOI.Nonnegatives(2n))
    return model
end

function jump_cvar(d)
    A0, c0 = d["A0"], vec(d["c0"])
    x_min, x_max = vec(d["x_min"]), vec(d["x_max"])
    gamma, kappa = _jscalar(d, "gamma"), _jscalar(d, "kappa")
    n_scen, nx = size(A0, 1), size(A0, 2)
    model = JuMP.Model()
    JuMP.@variable(model, x[1:nx])         # z = [x; alpha; u]
    JuMP.@variable(model, alpha)
    JuMP.@variable(model, u[1:n_scen])
    JuMP.@objective(model, Min, c0' * x)
    JuMP.@constraint(model,
        vcat(u .+ alpha .- A0 * x, u, kappa - alpha - gamma * sum(u),
             x .- x_min, x_max .- x) in MOI.Nonnegatives(2n_scen + 1 + 2nx))
    return model
end
