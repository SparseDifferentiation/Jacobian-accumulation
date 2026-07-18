# Worker CLI for the Julia sparse-AD canonicalization comparison.
# Invoked by canonicalization/julia_compare.py as a subprocess:
#
#   julia --project=julia --threads=1 canonicalization/julia/main.jl \
#       --problem SimpleQP --size full --data d.npz --warmup-data small.npz \
#       --out result.json [--dump-npz mats.npz] [--iters 3] [--resolves 20]
#
# The JSON result is (re)written after every phase so a parent-side timeout
# still leaves the completed phases on disk.  All matrices cross the process
# boundary as npz COO triplets (0-based indices), never JSON.

using DifferentiationInterface
using SparseConnectivityTracer
using SparseMatrixColorings
using ADTypes
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using SparseArrays
using LinearAlgebra
using Random
using Statistics
import JSON
import NPZ
import Pkg

LinearAlgebra.BLAS.set_num_threads(1)

include("problems.jl")
include("extract.jl")

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

function versions()
    deps = Pkg.dependencies()
    want = ("DifferentiationInterface", "SparseConnectivityTracer",
            "SparseMatrixColorings", "ForwardDiff", "ReverseDiff")
    d = Dict(info.name => string(info.version)
             for (_, info) in deps if info.name in want)
    d["julia"] = string(VERSION)
    return d
end

"One full extraction; returns (P, c, A, b, preps, stages)."
function extract_all(model; jacgrad::Bool)
    st = Dict{String,Any}()
    GC.gc()
    A, b, prepA, stA = extract_Ab(model.g, model.nz)
    merge!(st, stA)
    GC.gc()
    c, t_c = extract_c(model.f, model.nz)
    st["t_c_s"] = t_c
    GC.gc()
    P, prepP, stP = extract_P_native(model.f, model.nz)
    merge!(st, stP)
    st["single_native_s"] = st["t_prep_A_s"] + st["t_eval_A_s"] + t_c +
                            st["t_prep_P_s"] + st["t_eval_P_s"]
    if jacgrad
        try
            GC.gc()
            P2, t_jg = extract_P_jacgrad(model.f, model.nz)
            st["t_jacgrad_P_s"] = t_jg
            st["single_jacgrad_s"] = st["t_prep_A_s"] + st["t_eval_A_s"] + t_c + t_jg
            err = maximum(abs, P2 - P; init=0.0)
            st["jacgrad_vs_native_maxdiff"] = err
        catch e
            st["jacgrad_error"] = sprint(showerror, e)
        end
    end
    return P, c, A, b, (A=prepA, P=prepP), st
end

function build_model(name, data_path)
    d = NPZ.npzread(data_path)
    return BUILDERS[name](d)
end

function main()
    args = parse_args(ARGS)
    name, size_key = args["problem"], args["size"]
    iters = parse(Int, get(args, "iters", "3"))
    resolves = parse(Int, get(args, "resolves", "0"))
    out = args["out"]
    result = Dict{String,Any}(
        "problem" => name, "size" => size_key, "versions" => versions(),
    )
    flush_result() = open(io -> write(io, JSON.json(result)), out, "w")

    # -- warmup on the small instance: compiles every code path (JIT excluded
    #    from the target's numbers); ForwardDiff chunk-size specializations for
    #    the target size still land in the target's first iteration.
    if haskey(args, "warmup-data")
        wm = build_model(name, args["warmup-data"])
        wm.perturb_params! === nothing || wm.perturb_params!(MersenneTwister(0))
        extract_all(wm; jacgrad=true)
    end

    # -- cold singles
    t0 = time_ns()
    model = build_model(name, args["data"])
    result["build_s"] = (time_ns() - t0) / 1e9
    if model.set_params! !== nothing
        model.perturb_params!(MersenneTwister(7))   # θ values; overwritten below if verifying
    end
    if haskey(args, "theta")
        th = NPZ.npzread(args["theta"])
        model.set_params!([th[k] for k in sort(collect(keys(th)))])
    end

    P = c = A = b = nothing
    singles = Float64[]
    for it in 1:iters
        Pi, ci, Ai, bi, preps, st = extract_all(model; jacgrad=(it == 1))
        push!(singles, st["single_native_s"])
        if it == 1
            P, c, A, b = Pi, ci, Ai, bi
            result["stages_iter1"] = st
            result["single_iter1_s"] = st["single_native_s"]
        end
        result["singles_s"] = singles
        result["single_s"] = median(singles)
        flush_result()
    end

    # -- warm re-solves: refresh θ, redo only the θ-dependent extractions
    #    (A and b for ParametrizedQP; f has no parameters) with cached preps
    if resolves > 0 && model.perturb_params! !== nothing
        backend = sparse_fwd_backend()
        z0 = zeros(model.nz)
        prep = prepare_jacobian(model.g, backend, z0)
        rng = MersenneTwister(11)
        warm = Float64[]
        for _ in 1:resolves
            model.perturb_params!(rng)
            GC.gc()
            t0 = time_ns()
            jacobian(model.g, prep, backend, z0)
            model.g(z0)
            push!(warm, (time_ns() - t0) / 1e9)
        end
        result["resolves_s"] = warm
        result["resolve_s"] = median(warm)
        flush_result()
    end

    if haskey(args, "dump-npz")
        As = sparse(A)
        Ps = sparse(P)
        rA, cA, vA = findnz(As)
        rP, cP, vP = findnz(Ps)
        NPZ.npzwrite(args["dump-npz"], Dict(
            "A_rows" => rA .- 1, "A_cols" => cA .- 1, "A_vals" => vA,
            "A_shape" => collect(Base.size(As)),
            "P_rows" => rP .- 1, "P_cols" => cP .- 1, "P_vals" => vP,
            "P_shape" => collect(Base.size(Ps)),
            "b" => Vector{Float64}(b), "c" => Vector{Float64}(c),
        ))
        result["dumped"] = args["dump-npz"]
        flush_result()
    end
    println(JSON.json(result))
end

main()
