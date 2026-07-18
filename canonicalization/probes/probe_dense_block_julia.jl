# Does the Julia sparse-AD stack (SCT + SparseMatrixColorings +
# DifferentiationInterface/ForwardDiff) share CasADi's dense-block pathology?
#
# Mirror of probe_why_slow.py: all cases are affine g(x) = A*x with A constant,
# so the Jacobian *is* A and every stage's output is checkable exactly.
#   E1 dense tall block (m=2n), n doubling -- CasADi: one sweep per column,
#      16.3 s at 4000x2000.
#   E2 same shapes, sparse ~10 nnz/row -- CasADi collapses to 0.03 s.
#   E3 dense wide (n=2m) -- row/reverse compression territory.
#   E4 diagonal at LP scale -- linear regime.
# Matrix values are irrelevant to every stage (pattern-driven); Julia's own RNG
# replaces numpy's RandomState(0).
#
# Stages timed separately: pattern detection (SCT), coloring (SMC, both column
# and row partitions), DI preparation (re-runs detection+coloring internally --
# the tool's real cold cost), first eval, warm evals (median of 5).  Iteration
# order runs a tiny warmup case first so JIT compilation is excluded; ForwardDiff
# chunk-size recompilation per new size still lands in the first eval -- that is
# why first and warm evals are reported separately.
#
# Run:  julia --project=julia --threads=1 canonicalization/probes/probe_dense_block_julia.jl

using DifferentiationInterface
using SparseConnectivityTracer
using SparseMatrixColorings
using ADTypes
using ForwardDiff: ForwardDiff
using SparseArrays
using LinearAlgebra
using Random
using Statistics
using Printf
import JSON

LinearAlgebra.BLAS.set_num_threads(1)

const DETECTOR = TracerSparsityDetector()
const ALGO = GreedyColoringAlgorithm()

function probe(A, label; warm_evals=5)
    m, n = size(A)
    g(x) = A * x
    x0 = ones(n)

    t0 = time_ns()
    S = ADTypes.jacobian_sparsity(g, x0, DETECTOR)
    t_pattern = (time_ns() - t0) / 1e9

    t0 = time_ns()
    res_col = coloring(S, ColoringProblem(; structure=:nonsymmetric, partition=:column), ALGO)
    t_coloring = (time_ns() - t0) / 1e9
    res_row = coloring(S, ColoringProblem(; structure=:nonsymmetric, partition=:row), ALGO)
    nc_fwd, nc_rev = ncolors(res_col), ncolors(res_row)

    backend = AutoSparse(AutoForwardDiff(); sparsity_detector=DETECTOR, coloring_algorithm=ALGO)
    t0 = time_ns()
    prep = prepare_jacobian(g, backend, x0)
    t_prep = (time_ns() - t0) / 1e9

    t0 = time_ns()
    J = jacobian(g, prep, backend, x0)
    t_first = (time_ns() - t0) / 1e9

    warm = Float64[]
    for _ in 1:warm_evals
        t0 = time_ns()
        jacobian(g, prep, backend, x0)
        push!(warm, (time_ns() - t0) / 1e9)
    end
    t_warm = median(warm)

    # correctness: the Jacobian of x -> A*x is A, exactly
    @assert size(J) == (m, n)
    @assert isapprox(J, A; atol=1e-12, rtol=0) "J != A for $label"

    @printf("  %-34s pattern %8.3fs  color %7.3fs (fwd %5d rev %5d)  prep %8.3fs  eval1 %8.3fs  warm %8.4fs  nnz %d\n",
            label, t_pattern, t_coloring, nc_fwd, nc_rev, t_prep, t_first, t_warm, nnz(sparse(J)))
    return Dict(
        "label" => label, "m" => m, "n" => n, "nnz" => nnz(sparse(A)),
        "t_pattern_s" => t_pattern, "t_coloring_s" => t_coloring,
        "ncolors_fwd" => nc_fwd, "ncolors_rev" => nc_rev,
        "t_prep_s" => t_prep, "t_first_eval_s" => t_first, "t_warm_eval_s" => t_warm,
    )
end

function main()
    rng = MersenneTwister(0)
    results = Dict{String,Any}[]

    # JIT warmup (excluded from results): tiny dense + sparse cases compile all paths
    probe(randn(rng, 32, 16), "warmup dense 32x16")
    probe(sprandn(rng, 32, 16, 0.3), "warmup sparse 32x16")
    println("  --- warmup done, results below ---")

    println("E1: dense tall block (m=2n), n doubling")
    for n in (250, 500, 1000, 2000)
        push!(results, probe(randn(rng, 2n, n), "dense $(2n)x$(n)"))
    end

    println("E2: same shapes, sparse ~10 nnz/row")
    for n in (250, 500, 1000, 2000)
        push!(results, probe(sprandn(rng, 2n, n, 10.0 / n), "sparse $(2n)x$(n) (10/row)"))
    end

    println("E3: dense wide block (n=2m), m doubling")
    for m in (250, 500, 1000, 2000)
        push!(results, probe(randn(rng, m, 2m), "dense $(m)x$(2m)"))
    end

    println("E4: diagonal, LP scale (linear regime)")
    for n in (10^5, 10^6, 4 * 10^6)
        push!(results, probe(sparse(1.0I, n, n), "diag $n"; warm_evals=3))
    end

    out = joinpath(@__DIR__, "..", "results_probe_dense_block_julia.json")
    open(out, "w") do io
        write(io, JSON.json(Dict(
            "julia" => string(VERSION),
            "packages" => Dict("DifferentiationInterface" => "0.7.20",
                               "SparseConnectivityTracer" => "1.2.2",
                               "SparseMatrixColorings" => "0.4.27",
                               "ForwardDiff" => "1.4.1"),
            "results" => results,
        )))
    end
    println("wrote $out")
end

main()
