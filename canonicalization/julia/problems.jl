# Lowered problem models for the Julia sparse-AD comparison.
#
# Each builder takes the npz constant dict written by julia_compare.py and
# returns a NamedTuple
#   (f, g, nz, set_params!, perturb_params!)
# where f(z) is the lowered objective, g(z) = b - A z the slack expression
# (same z layout and row order as the CasADi Specs -- see _lowered_data.py),
# nz the number of columns, set_params!(vals) installs parameter values in
# place, and perturb_params!(rng) draws fresh values for warm re-solves
# (nothing for non-parametric problems).

_scalar(d, k) = (v = d[k]; v isa AbstractArray ? first(v) : v)

function build_simple_qp(d)
    P0, q0 = d["P0"], vec(d["q0"])
    G0, h0 = d["G0"], vec(d["h0"])
    Aeq, beq = d["Aeq"], vec(d["beq"])
    n = length(q0)
    f(z) = 0.5 * LinearAlgebra.dot(z, P0 * z) + LinearAlgebra.dot(q0, z)
    g(z) = vcat(beq - Aeq * z, h0 - G0 * z)   # zero cone rows, then nonneg
    return (f=f, g=g, nz=n, set_params! = nothing, perturb_params! = nothing)
end

function build_least_squares(d)
    A0, b0 = d["A0"], vec(d["b0"])
    m, n = size(A0)
    # z = [t; x];  min t't  s.t.  b0 + t - A0 x  in zero cone
    f(z) = LinearAlgebra.dot(view(z, 1:m), view(z, 1:m))
    g(z) = b0 .+ z[1:m] .- A0 * z[m+1:m+n]
    return (f=f, g=g, nz=m + n, set_params! = nothing, perturb_params! = nothing)
end

function build_parametrized_qp(d)
    m, n = Int(_scalar(d, "m")), Int(_scalar(d, "n"))
    Ap = zeros(m, n)          # parameter storage, mutated in place
    bp = zeros(m)
    # z = [t; x];  rows: zero (b + t - A x), nonneg (x, 1 - x)
    f(z) = LinearAlgebra.dot(view(z, 1:m), view(z, 1:m))
    function g(z)
        t, x = z[1:m], z[m+1:m+n]
        vcat(bp .+ t .- Ap * x, x, 1 .- x)
    end
    set_params!(vals) = (Ap .= vals[1]; bp .= vec(vals[2]); nothing)
    perturb_params!(rng) = (Random.randn!(rng, Ap); Random.randn!(rng, bp); nothing)
    return (f=f, g=g, nz=m + n, set_params! = set_params!, perturb_params! = perturb_params!)
end

function build_cvar(d)
    A0, c0 = d["A0"], vec(d["c0"])
    x_min, x_max = vec(d["x_min"]), vec(d["x_max"])
    gamma, kappa = _scalar(d, "gamma"), _scalar(d, "kappa")
    n_scen, nx = size(A0, 1), size(A0, 2)
    # z = [x; alpha; u]; rows (all nonneg):
    #   u + alpha - A x,  u,  kappa - alpha - gamma sum(u),  x - xmin,  xmax - x
    f(z) = LinearAlgebra.dot(c0, view(z, 1:nx))
    function g(z)
        x = z[1:nx]
        alpha = z[nx+1]
        u = z[nx+2:nx+1+n_scen]
        vcat(u .+ alpha .- A0 * x, u, kappa - alpha - gamma * sum(u),
             x .- x_min, x_max .- x)
    end
    return (f=f, g=g, nz=nx + 1 + n_scen, set_params! = nothing, perturb_params! = nothing)
end

const BUILDERS = Dict(
    "SimpleQP" => build_simple_qp,
    "LeastSquares" => build_least_squares,
    "ParametrizedQP" => build_parametrized_qp,
    "CVaRSlice" => build_cvar,
)
