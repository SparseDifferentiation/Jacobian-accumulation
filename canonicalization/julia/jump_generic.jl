# Generic JuMP/MOI worker for the full-suite comparison, driven by the
# tool-neutral lowered blocks of _lowered_blocks.py (passed as npz).
#
#   julia --project=julia --threads=1 canonicalization/julia/jump_generic.jl \
#       --data blocks.npz --out result.json [--warmup-data small.npz] \
#       [--dump-npz mats.npz] [--iters 3] [--resolves 20]
#
# Model: variables z in block column order; per row block a vector constraint
#   (b - A z) in K   with K in {Zeros, Nonnegatives, SecondOrderCone blocks,
#   Scaled PSD triangle}; objective 0.5 z'Pz + c'z.  Cone tags are metadata:
# copy_to's MatrixOfConstraints assembly is set-agnostic, and the product-of-
# sets groups rows Zeros -> Nonnegatives -> SOC -> PSD, i.e. CVXPY's CLARABEL
# row order.  (MOI's scaled-PSD svec sequence -- column-major upper triangle,
# off-diagonals * sqrt2 -- coincides entrywise with CVXPY's row-major lower
# order for symmetric data.)
#
# Timed "single" = MOI.copy_to into the matrix cache (JuMP's canonicalization
# analog).  Model construction from the block arrays is build_s: reported,
# excluded, exactly like casadi_build_s.

using JuMP
import MathOptInterface as MOI
const MOIU = MOI.Utilities
using SparseArrays
using LinearAlgebra
using Statistics
import JSON
import NPZ

LinearAlgebra.BLAS.set_num_threads(1)

MOIU.@product_of_sets(
    ClarabelCones,
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.SecondOrderCone,
    MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle},
)
const CacheModel = MOIU.GenericModel{
    Float64,
    MOIU.ObjectiveContainer{Float64},
    MOIU.VariablesContainer{Float64},
    MOIU.MatrixOfConstraints{
        Float64,
        MOIU.MutableSparseMatrixCSC{Float64,Int,MOIU.OneBasedIndexing},
        Vector{Float64},
        ClarabelCones{Float64},
    },
}

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

function load_blocks(path)
    d = NPZ.npzread(path)
    nz = Int(d["nz"] isa AbstractArray ? first(d["nz"]) : d["nz"])
    nb = Int(d["nblocks"] isa AbstractArray ? first(d["nblocks"]) : d["nblocks"])
    blocks = NamedTuple[]
    for k in 0:(nb - 1)
        m = Int(d["rb$(k)_m"] isa AbstractArray ? first(d["rb$(k)_m"]) : d["rb$(k)_m"])
        A = sparse(Int.(d["rb$(k)_rows"]) .+ 1, Int.(d["rb$(k)_cols"]) .+ 1,
                   Vector{Float64}(d["rb$(k)_vals"]), m, nz)
        kind = Int(d["rb$(k)_kind"] isa AbstractArray ? first(d["rb$(k)_kind"]) : d["rb$(k)_kind"])
        dims = haskey(d, "rb$(k)_dims") ? Int.(vec(d["rb$(k)_dims"])) : Int[]
        side = haskey(d, "rb$(k)_side") ?
            Int(d["rb$(k)_side"] isa AbstractArray ? first(d["rb$(k)_side"]) : d["rb$(k)_side"]) : 0
        push!(blocks, (kind=kind, A=A, b=Vector{Float64}(vec(d["rb$(k)_b"])),
                       dims=dims, side=side))
    end
    P = haskey(d, "P_rows") ?
        sparse(Int.(d["P_rows"]) .+ 1, Int.(d["P_cols"]) .+ 1,
               Vector{Float64}(d["P_vals"]), nz, nz) : spzeros(nz, nz)
    return (nz=nz, blocks=blocks, P=P, c=Vector{Float64}(vec(d["c"])))
end

function build_model(pb)
    model = JuMP.Model()
    JuMP.@variable(model, z[1:pb.nz])
    for blk in pb.blocks
        expr = blk.b - blk.A * z
        if blk.kind == 0
            JuMP.@constraint(model, expr in MOI.Zeros(length(blk.b)))
        elseif blk.kind == 1
            JuMP.@constraint(model, expr in MOI.Nonnegatives(length(blk.b)))
        elseif blk.kind == 2
            lo = 1
            for dsz in blk.dims
                JuMP.@constraint(model, expr[lo:lo+dsz-1] in MOI.SecondOrderCone(dsz))
                lo += dsz
            end
        elseif blk.kind == 3
            JuMP.@constraint(
                model, expr in MOI.Scaled(MOI.PositiveSemidefiniteConeTriangle(blk.side)))
        else
            error("unknown kind $(blk.kind)")
        end
    end
    if nnz(pb.P) > 0
        JuMP.@objective(model, Min, 0.5 * (z' * pb.P * z) + pb.c' * z)
    else
        JuMP.@objective(model, Min, pb.c' * z)
    end
    return model
end

function canonicalize(model)
    src = JuMP.backend(model)
    dest = CacheModel()
    GC.gc()
    t0 = time_ns()
    index_map = MOI.copy_to(dest, src)
    t = (time_ns() - t0) / 1e9
    return dest, index_map, t
end

function extract_matrices(model, dest, index_map)
    src = JuMP.backend(model)
    vis = MOI.get(src, MOI.ListOfVariableIndices())
    nz = length(vis)
    col = [index_map[vi].value for vi in vis]
    perm = invperm(col)

    M = convert(SparseMatrixCSC{Float64,Int}, dest.constraints.coefficients)
    A = (-M)[:, perm]
    b = copy(dest.constraints.constants)

    P = spzeros(nz, nz)
    c = zeros(nz)
    F = MOI.get(dest, MOI.ObjectiveFunctionType())
    f = MOI.get(dest, MOI.ObjectiveFunction{F}())
    if F <: MOI.ScalarQuadraticFunction
        for t in f.quadratic_terms
            i, j = perm[t.variable_1.value], perm[t.variable_2.value]
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
    iters = parse(Int, get(args, "iters", "3"))
    resolves = parse(Int, get(args, "resolves", "0"))
    out = args["out"]
    result = Dict{String,Any}(
        "versions" => Dict("julia" => string(VERSION),
                           "JuMP" => string(pkgversion(JuMP)),
                           "MathOptInterface" => string(pkgversion(MOI))),
    )
    flush_result() = open(io -> write(io, JSON.json(result)), out, "w")

    if haskey(args, "warmup-data")
        pw = load_blocks(args["warmup-data"])
        mw = build_model(pw)
        dw, iw, _ = canonicalize(mw)
        extract_matrices(mw, dw, iw)
    end

    t0 = time_ns()
    pb = load_blocks(args["data"])
    result["blocks_load_s"] = (time_ns() - t0) / 1e9

    builds = Float64[]
    singles = Float64[]
    model = nothing
    dest = index_map = nothing
    for it in 1:iters
        t0 = time_ns()
        model = build_model(pb)
        push!(builds, (time_ns() - t0) / 1e9)
        dest, index_map, t = canonicalize(model)
        push!(singles, t)
        result["build_s"] = median(builds)
        result["singles_s"] = singles
        result["single_s"] = median(singles)
        result["single_iter1_s"] = singles[1]
        flush_result()
    end

    if resolves > 0
        warm = Float64[]
        for _ in 1:resolves
            _, _, t = canonicalize(model)
            push!(warm, t)
        end
        result["resolve_s"] = median(warm)
        result["resolve_note"] = "fresh copy_to on the built model (no vectorized parameter refresh in base JuMP; see ParametricOptInterface.jl)"
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
