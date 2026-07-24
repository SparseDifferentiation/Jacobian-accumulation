# End-to-end JuMP worker: user model (models/<Name>.jl) -> SCS handoff.
#
#   julia --project=julia --threads=1 canonicalization/jump_e2e/runner.jl \
#       --problem SimpleQP --data d.npz --out result.json [--theta t.npz] \
#       [--warmup-data w.npz] [--warmup-theta wt.npz] [--iters 3] \
#       [--mode handoff|solve] [--eps 1e-4]
#
# Default mode "handoff" NEVER RUNS SCS.  Per iteration, two timers:
#   build_s    -- the JuMP macro layer: raw arrays -> model objects (the
#                 honest user-model construction cost; where JuMP's
#                 coefficient assembly happens);
#   handoff_s  -- MOI.instantiate(SCS.Optimizer, bridged) + MOI.copy_to:
#                 bridging + assembly into SCS.jl's own input cache (a
#                 zero-based-CSC MatrixOfConstraints in SCS cone order) --
#                 the moment the solver COULD start, i.e. JuMP's analog of
#                 CVXPY's get_problem_data(solver=SCS).
#
# Mode "solve" (opt-in, used by compare.py --verify only) additionally calls
# optimize! and reports objective/status/solver time, so model equivalence
# can be gated on optimal values.
#
# Warmup runs the identical path on the small instance first so Julia JIT is
# excluded (the same rule as the other Julia workers).

using JuMP
import SCS
import MathOptInterface as MOI
using LinearAlgebra
using Statistics
import JSON
import NPZ

LinearAlgebra.BLAS.set_num_threads(1)

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

function load_theta(path)
    th = NPZ.npzread(path)
    return [th[k] for k in sort(collect(keys(th)))]   # theta_0, theta_1, ...
end

function run_handoff(d, θ)
    GC.gc()
    t0 = time_ns()
    model = build(d, θ)
    t_build = (time_ns() - t0) / 1e9

    t0 = time_ns()
    dest = MOI.instantiate(SCS.Optimizer; with_bridge_type = Float64)
    MOI.set(dest, MOI.Silent(), true)
    MOI.copy_to(dest, backend(model))
    t_handoff = (time_ns() - t0) / 1e9
    return t_build, t_handoff
end

function run_solve(d, θ, eps)
    GC.gc()
    t0 = time_ns()
    model = build(d, θ)
    t_build = (time_ns() - t0) / 1e9

    t0 = time_ns()
    set_optimizer(model, SCS.Optimizer)
    set_attribute(model, "eps_abs", eps)
    set_attribute(model, "eps_rel", eps)
    set_silent(model)
    optimize!(model)
    t_solve = (time_ns() - t0) / 1e9

    status = string(termination_status(model))
    obj = try objective_value(model) catch; NaN end
    scs_t = try MOI.get(model, MOI.SolveTimeSec()) catch; NaN end
    return t_build, t_solve, obj, status, scs_t
end

function main(args)
    name = args["problem"]
    iters = parse(Int, get(args, "iters", "3"))
    eps = parse(Float64, get(args, "eps", "1e-4"))
    mode = get(args, "mode", "handoff")
    out = args["out"]

    d = NPZ.npzread(args["data"])
    θ = haskey(args, "theta") ? load_theta(args["theta"]) : nothing

    result = Dict{String,Any}(
        "problem" => name, "eps" => eps, "mode" => mode,
        "versions" => Dict("julia" => string(VERSION),
                           "JuMP" => string(pkgversion(JuMP)),
                           "MathOptInterface" => string(pkgversion(MOI)),
                           "SCS" => string(pkgversion(SCS))),
    )
    flush_result() = open(io -> write(io, JSON.json(result)), out, "w")

    if haskey(args, "warmup-data")
        dw = NPZ.npzread(args["warmup-data"])
        θw = haskey(args, "warmup-theta") ? load_theta(args["warmup-theta"]) : nothing
        mode == "handoff" ? run_handoff(dw, θw) : run_solve(dw, θw, eps)
    end

    if mode == "handoff"
        builds, handoffs = Float64[], Float64[]
        for _ in 1:iters
            t_build, t_handoff = run_handoff(d, θ)
            push!(builds, t_build); push!(handoffs, t_handoff)
            result["build_s"] = median(builds)
            result["handoff_s"] = median(handoffs)
            result["e2e_s"] = median(builds .+ handoffs)
            result["builds_s"] = builds
            result["handoffs_s"] = handoffs
            flush_result()
        end
    else
        builds, solves, scs_ts = Float64[], Float64[], Float64[]
        for _ in 1:iters
            t_build, t_solve, obj, status, scs_t = run_solve(d, θ, eps)
            push!(builds, t_build); push!(solves, t_solve); push!(scs_ts, scs_t)
            result["objective"] = obj
            result["status"] = status
            result["build_s"] = median(builds)
            result["solve_s"] = median(solves)
            finite = filter(isfinite, scs_ts)
            result["scs_solve_s"] = isempty(finite) ? nothing : median(finite)
            result["e2e_s"] = median(builds .+ solves)
            result["builds_s"] = builds
            result["solves_s"] = solves
            flush_result()
        end
    end
    println(JSON.json(result))
end

# model include must happen at top level (before main runs) so the freshly
# defined build(d, θ) is callable without world-age gymnastics
const ARGS_D = parse_args(ARGS)
include(joinpath(@__DIR__, "models", ARGS_D["problem"] * ".jl"))
main(ARGS_D)
