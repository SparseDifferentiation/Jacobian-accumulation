# Worker CLI for the JuMP/MOI canonicalization comparison.
#
#   julia --project=julia --threads=1 canonicalization/julia/jump_main.jl \
#       --problem SimpleQP --size full --data d.npz --out result.json \
#       [--dump-npz mats.npz] [--iters 3] [--theta theta.npz]
#
# What is timed ("single"): MOI.copy_to from JuMP's dict-form model cache into
# a MatrixOfConstraints-backed cache -- the step where the sparse constraint
# matrix A (CSC) and the objective data (P, q) are materialized.  This is
# byte-for-byte the machinery matrix-form solver wrappers (Clarabel.jl, SCS.jl,
# ...) run inside MOI.Utilities.attach_optimizer: JuMP's canonicalization
# analog.  Model construction (JuMP macros -> dict cache) is the
# graph-construction analog: excluded from the headline, reported as build_s.
#
# No AD, no sparsity detection, no coloring happens anywhere in this path.
#
# Warm re-solves: base JuMP has no vectorized parameter refresh (MOI.modify is
# per-coefficient; ParametricOptInterface.jl is the ecosystem's answer) -- the
# honest warm path here is a fresh copy_to, reported as resolve_s.

using JuMP
import MathOptInterface as MOI
const MOIU = MOI.Utilities
using SparseArrays
using LinearAlgebra
using Statistics
import JSON
import NPZ
import Pkg

LinearAlgebra.BLAS.set_num_threads(1)

include("jump_problems.jl")

# Clarabel-form target cache: rows grouped Zeros-then-Nonnegatives, CSC storage.
MOIU.@product_of_sets(ZerosNonneg, MOI.Zeros, MOI.Nonnegatives)
const CacheModel = MOIU.GenericModel{
    Float64,
    MOIU.ObjectiveContainer{Float64},
    MOIU.VariablesContainer{Float64},
    MOIU.MatrixOfConstraints{
        Float64,
        MOIU.MutableSparseMatrixCSC{Float64,Int,MOIU.OneBasedIndexing},
        Vector{Float64},
        ZerosNonneg{Float64},
    },
}

const JUMP_BUILDERS = Dict(
    "SimpleQP" => (d, θ) -> jump_simple_qp(d),
    "LeastSquares" => (d, θ) -> jump_least_squares(d),
    "ParametrizedQP" => (d, θ) -> jump_parametrized_qp(d, θ[1], vec(θ[2])),
    "CVaRSlice" => (d, θ) -> jump_cvar(d),
)

function parse_args(argv)
    args = Dict{String,String}()
    i = 1
    while i <= length(argv)
        startswith(argv[i], "--") || error("bad arg $(argv[i])")
        args[argv[i][3:end]] = argv[i+1]
        i += 2
    end
    return args
end

"copy_to into the matrix-form cache; returns (dest, index_map, seconds)."
function canonicalize(model)
    src = JuMP.backend(model)
    dest = CacheModel()
    GC.gc()
    t0 = time_ns()
    index_map = MOI.copy_to(dest, src)
    t = (time_ns() - t0) / 1e9
    return dest, index_map, t
end

"Recover (P, c, A, b) in the SOURCE model's variable order from the cache."
function extract_matrices(model, dest, index_map)
    src = JuMP.backend(model)
    vis = MOI.get(src, MOI.ListOfVariableIndices())      # creation order
    nz = length(vis)
    col = [index_map[vi].value for vi in vis]            # dest column of src var i
    perm = invperm(col)                                   # dest col -> src order

    M = convert(SparseMatrixCSC{Float64,Int}, dest.constraints.coefficients)
    consts = copy(dest.constraints.constants)
    # rows are stored as  M z + const in K;  Clarabel/CVXPY form is  b - A z in K
    A = (-M)[:, perm]
    b = consts

    # objective: P (may be absent) and c, in source variable order
    P = spzeros(nz, nz)
    c = zeros(nz)
    F = MOI.get(dest, MOI.ObjectiveFunctionType())
    f = MOI.get(dest, MOI.ObjectiveFunction{F}())
    if F <: MOI.ScalarQuadraticFunction
        for t in f.quadratic_terms
            i, j = perm[t.variable_1.value], perm[t.variable_2.value]
            # MOI stores 0.5*z'Qz via terms with the diagonal doubled;
            # ScalarQuadraticTerm(c,i,j) contributes c to Q[i,j] (and Q[j,i])
            P[i, j] += t.coefficient
            if i != j
                P[j, i] += t.coefficient
            end
        end
        for t in f.affine_terms
            c[perm[t.variable.value]] += t.coefficient
        end
    elseif F <: MOI.ScalarAffineFunction
        for t in f.terms
            c[perm[t.variable.value]] += t.coefficient
        end
    end
    return P, c, A, b
end

function main()
    args = parse_args(ARGS)
    name = args["problem"]
    iters = parse(Int, get(args, "iters", "3"))
    resolves = parse(Int, get(args, "resolves", "0"))
    out = args["out"]
    d = NPZ.npzread(args["data"])
    θ = nothing
    if haskey(args, "theta")
        th = NPZ.npzread(args["theta"])
        θ = [th[k] for k in sort(collect(keys(th)))]
    end

    result = Dict{String,Any}(
        "problem" => name, "size" => args["size"],
        "versions" => Dict("julia" => string(VERSION),
                           "JuMP" => string(pkgversion(JuMP)),
                           "MathOptInterface" => string(pkgversion(MOI))),
    )
    flush_result() = open(io -> write(io, JSON.json(result)), out, "w")

    # warmup on the small instance: JIT for the whole path
    if haskey(args, "warmup-data")
        dw = NPZ.npzread(args["warmup-data"])
        θw = nothing
        if name == "ParametrizedQP"
            mw_, nw_ = Int(_jscalar(dw, "m")), Int(_jscalar(dw, "n"))
            θw = [randn(mw_, nw_), randn(mw_)]
        end
        mw = JUMP_BUILDERS[name](dw, θw)
        destw, imw, _ = canonicalize(mw)
        extract_matrices(mw, destw, imw)
    end

    builds = Float64[]
    singles = Float64[]
    model = nothing
    dest = index_map = nothing
    for it in 1:iters
        t0 = time_ns()
        model = JUMP_BUILDERS[name](d, θ)
        push!(builds, (time_ns() - t0) / 1e9)
        dest, index_map, t = canonicalize(model)
        push!(singles, t)
        result["build_s"] = median(builds)
        result["singles_s"] = singles
        result["single_s"] = median(singles)
        result["single_iter1_s"] = singles[1]
        flush_result()
    end

    # warm path = fresh copy_to on the existing model (no vectorized modify)
    if resolves > 0
        warm = Float64[]
        for _ in 1:resolves
            _, _, t = canonicalize(model)
            push!(warm, t)
        end
        result["resolve_s"] = median(warm)
        result["resolve_note"] = "fresh copy_to; base JuMP has no vectorized parameter refresh (see ParametricOptInterface.jl)"
        flush_result()
    end

    if haskey(args, "dump-npz")
        P, c, A, b = extract_matrices(model, dest, index_map)
        rA, cA, vA = findnz(sparse(A))
        rP, cP, vP = findnz(sparse(P))
        NPZ.npzwrite(args["dump-npz"], Dict(
            "A_rows" => rA .- 1, "A_cols" => cA .- 1, "A_vals" => vA,
            "A_shape" => collect(size(A)),
            "P_rows" => rP .- 1, "P_cols" => cP .- 1, "P_vals" => vP,
            "P_shape" => collect(size(P)),
            "b" => Vector{Float64}(b), "c" => Vector{Float64}(c),
        ))
        result["dumped"] = args["dump-npz"]
        flush_result()
    end
    println(JSON.json(result))
end

main()
