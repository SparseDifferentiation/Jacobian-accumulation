# =============================================================================
# Sparse Jacobian computation with the Julia sparse-AD stack:
#   SparseConnectivityTracer.jl (detection) + SparseMatrixColorings.jl (coloring)
#   + DifferentiationInterface.jl (compressed accumulation over ForwardDiff).
# =============================================================================
# Same tridiagonal system as the casadi/jax/diffengine/adolc examples:
# f_i = sin(x_i) + x_{i-1}*x_i + exp(-x_i*x_{i+1}), so each f_i depends only on
# x[i-1], x[i], x[i+1] and the Jacobian is sparse.
#
# Run:  julia --project=julia --threads=1 jacobian/sct_tridiag_jacobian.jl

using DifferentiationInterface
using SparseConnectivityTracer
using SparseMatrixColorings
using ADTypes
using ForwardDiff: ForwardDiff
using SparseArrays
using LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)

n = 10

function f(x)
    n = length(x)
    [sin(x[i]) + (i > 1 ? x[i-1] * x[i] : 0.0) +
     (i < n ? exp(-x[i] * x[i+1]) : 0.0) for i in 1:n]
end

# =============================================================================
# Sparsity detection — SparseConnectivityTracer
# =============================================================================
# Unlike CasADi's bitvector sweeps over a built graph, SCT runs f ONCE on a
# tracer type that propagates index sets through every operation ("binarization
# of the chain rule").  The result is a global (input-independent) pattern.

detector = TracerSparsityDetector()
S = ADTypes.jacobian_sparsity(f, ones(n), detector)
println("Jacobian size: ", size(S, 1), " x ", size(S, 2))
println("Non-zeros: ", nnz(S), " out of ", length(S), " elements")
println("\nSparsity pattern (1 = nonzero):")
show(stdout, MIME"text/plain"(), Int.(Matrix(S)))
println()

# =============================================================================
# Graph coloring — SparseMatrixColorings
# =============================================================================
# Same idea as CasADi's uni_coloring: structurally orthogonal columns share a
# color and are computed in one forward pass.  Tridiagonal bandwidth 3 => 3
# colors regardless of n.

problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
algo = GreedyColoringAlgorithm()
result = coloring(S, problem, algo)
println("\nGraph coloring: ", ncolors(result), " colors (forward passes) vs ",
        n, " without coloring")
println("Column colors: ", column_colors(result))

# =============================================================================
# Compressed accumulation — DifferentiationInterface
# =============================================================================
# AutoSparse wraps ForwardDiff: detection + coloring happen once in
# prepare_jacobian; each jacobian() call then does ncolors compressed forward
# passes and decompresses into the sparse matrix.

backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=detector,
    coloring_algorithm=algo,
)
x0 = ones(n)
prep = prepare_jacobian(f, backend, x0)
J = jacobian(f, prep, backend, x0)

println("\nNumerical Jacobian at x = ones(10):")
show(stdout, MIME"text/plain"(), Matrix(J))
println()
