# (P, c, A, b) extraction through the generic Julia sparse-AD pipeline:
# SparseConnectivityTracer (detection) + SparseMatrixColorings (coloring) +
# DifferentiationInterface (compressed accumulation).
#
# Mirrors the CasADi harness formulas:
#   A = -sparse_jacobian(g) at z=0        b = g(0)
#   c = gradient(f) at z=0                P = sparse_hessian(f)
# P is attempted along two routes, mirroring the harness's jacobian-of-gradient
# fairness rule for CasADi:
#   native  -- DI sparse Hessian (HessianTracer pattern + symmetric coloring +
#              forward-over-reverse HVPs); the stack's documented path.
#   jacgrad -- sparse Jacobian (SCT+ForwardDiff) of z -> gradient(f)(z) with the
#              gradient taken by ReverseDiff; only valid when SCT tracers can
#              flow through the ReverseDiff tape, so failures are recorded, not
#              fatal.
# Every stage is timed separately; "prep" repeats detection+coloring internally
# (DI's real cold path), so t_pattern/t_coloring are the decomposition, not
# additional cost.

const DETECTOR = TracerSparsityDetector()
const COLORALG = GreedyColoringAlgorithm()

sparse_fwd_backend() =
    AutoSparse(AutoForwardDiff(); sparsity_detector=DETECTOR, coloring_algorithm=COLORALG)

hessian_backend() = AutoSparse(
    SecondOrder(AutoForwardDiff(), AutoReverseDiff());
    sparsity_detector=DETECTOR, coloring_algorithm=COLORALG,
)

"""Sparse Jacobian of g at z=0 with per-stage timers.  Returns (A = -J, b, prep, stages)."""
function extract_Ab(g, nz)
    z0 = zeros(nz)
    st = Dict{String,Any}()

    t0 = time_ns()
    S = ADTypes.jacobian_sparsity(g, z0, DETECTOR)
    st["t_pattern_s"] = (time_ns() - t0) / 1e9
    st["nnz_A"] = SparseArrays.nnz(S)

    t0 = time_ns()
    res = coloring(S, ColoringProblem(; structure=:nonsymmetric, partition=:column), COLORALG)
    st["t_coloring_s"] = (time_ns() - t0) / 1e9
    st["ncolors_A"] = ncolors(res)

    backend = sparse_fwd_backend()
    t0 = time_ns()
    prep = prepare_jacobian(g, backend, z0)
    st["t_prep_A_s"] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    J = jacobian(g, prep, backend, z0)
    st["t_eval_A_s"] = (time_ns() - t0) / 1e9

    b = g(z0)
    return -J, b, (prep=prep, backend=backend), st
end

"""Gradient of f at z=0 (dense reverse mode)."""
function extract_c(f, nz)
    z0 = zeros(nz)
    t0 = time_ns()
    c = DifferentiationInterface.gradient(f, AutoReverseDiff(), z0)
    return c, (time_ns() - t0) / 1e9
end

"""Sparse Hessian of f, native DI path."""
function extract_P_native(f, nz)
    z0 = zeros(nz)
    st = Dict{String,Any}()
    backend = hessian_backend()
    t0 = time_ns()
    prep = prepare_hessian(f, backend, z0)
    st["t_prep_P_s"] = (time_ns() - t0) / 1e9
    t0 = time_ns()
    P = hessian(f, prep, backend, z0)
    st["t_eval_P_s"] = (time_ns() - t0) / 1e9
    st["nnz_P"] = SparseArrays.nnz(SparseArrays.sparse(P))
    return P, (prep=prep, backend=backend), st
end

"""Sparse Hessian as sparse Jacobian of the ReverseDiff gradient (may fail)."""
function extract_P_jacgrad(f, nz)
    z0 = zeros(nz)
    gradf(z) = DifferentiationInterface.gradient(f, AutoReverseDiff(), z)
    backend = sparse_fwd_backend()
    t0 = time_ns()
    prep = prepare_jacobian(gradf, backend, z0)
    P = jacobian(gradf, prep, backend, z0)
    return P, (time_ns() - t0) / 1e9
end
