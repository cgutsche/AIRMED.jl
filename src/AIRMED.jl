"""
AIRMED: Adaptive Improvement and Realignment of simulation ModEls for Digital twins

A Julia framework for autonomous digital twin simulation with continuous model
validation, ML-based adaptation, and agent-in-the-loop supervision.
"""
module AIRMED

using ModelingToolkit
using OrdinaryDiffEq
using Lux
using ComponentArrays
using SciMLSensitivity
using Optimization
using OptimizationOptimisers
# `import` (not `using`): OptimizationOptimJL re-exports Optim, whose `Adam`
# would clash with the Optimisers.Adam brought in by OptimizationOptimisers.
import OptimizationOptimJL
using Statistics
using LinearAlgebra
using Random
using CSV
using DataFrames
using Dates
using HTTP
using JSON3

include("types.jl")
include("model.jl")
include("simulation.jl")
include("validation.jl")
include("ml_update.jl")
include("agent_supervisor.jl")
include("model_adaptation.jl")
include("twin_rebuild.jl")

export AIRMEDProblem
export ComponentHook, ComponentGuess, DriftConfig, SupervisionConfig
export DriftDetectionMethod, CUSUM, EWMA, SimpleThreshold
export HookResidualSpec
export CSVDataSource, FunctionDataSource
export DriftResult, UpdateResult
export ComponentProposal, ModelAdaptationResult, LLMExchange
export AdaptableParam, ParameterAdjustment
export AgentState

export simulate, simulate_ude, load_data, interpolate_to_times
export compute_residuals, detect_drift
export build_nn, train_ude, train_symbolic_ude, symbolic_regression_of_nn, run_update_workflow
export sparse_regression
export model_update_summary
export propose_model_adaptation, model_adaptation_summary
export build_adapted_system, build_adapted_problem, build_parameter_adjusted_problem
export build_fix_problem, rebuild_problem
export run_airmed
export agent_supervise!, record_update!, record_adaptation!, generate_report
export explain_for_user
export model_summary, get_state_names, get_param_names, make_ode_problem
export add_nn_to_ode

"""
    run_airmed(problem, base_ode!, nn_input_fn; kwargs...)
        -> (AgentState, Union{UpdateResult, Nothing}, Union{ModelAdaptationResult, Nothing})

Run one iteration of the workflow: load data, simulate, compute residuals,
detect drift, ask the supervisor, then optionally train a UDE and propose a
structural adaptation.

The network correction is diagnostic only. The operating twin remains the
symbolic MTK model, base or adapted.

# Arguments
- `problem`: the problem to check.
- `base_ode!`: `(du, u, p_base, t)`, the known physics in plain Julia, so that
  it can be differentiated during UDE training.
- `nn_input_fn`: `(u, t) -> Vector`, the network inputs.

# Keywords
- `n_points`: samples per window. Default `200`.
- `verbose`: print progress. Default `true`.
- `nn_input_dim`, `nn_output_dim`, `nn_hidden`, `nn_depth`: network shape.
- `train_ude`: train a UDE when the supervisor decides `:update`. Default
  `false`, so the adaptation runs on hook-local and sensor evidence alone
  and the returned `UpdateResult` slot stays `nothing`.
- `adapt`: run `propose_model_adaptation` afterwards. Default `false`.
- `adapt_api`, `adapt_api_key`, `adapt_base_url`, `adapt_model`,
  `adapt_num_ctx`, `adapt_think`: LLM settings for that step. `nothing` takes
  the `agent_*` field of `problem.supervision_config`.
- `adapt_input_ranges`: ranges for the network characterisation. `nothing`
  derives them from the fitted trajectory.
- `adapt_min_drift_explained`: skip the adaptation when a trained UDE explained
  less of the drift than this. Default `0.25`, `0` disables the check.
- `adapt_kwargs`: forwarded to `propose_model_adaptation`, e.g.
  `(; max_retries = 5)`.
- `state`: reuse a previous `AgentState` to accumulate history across calls.
- `train_kwargs...`: forwarded to `train_ude`.

# Returns
`(state, update_result, adaptation_result)`. The second is `nothing` without
UDE training or without an update, the third unless the adaptation ran.
Two-variable destructuring remains valid.
"""
function run_airmed(
    problem::AIRMEDProblem,
    base_ode!::Function,
    nn_input_fn::Function;
    n_points::Int     = 200,
    verbose::Bool     = true,
    nn_input_dim::Int = 2,
    nn_output_dim     = nothing,
    nn_hidden::Int    = 16,
    nn_depth::Int     = 2,
    adapt::Bool       = false,
    # With `false`, UDE training and the update bookkeeping are skipped; the
    # adaptation then runs on hook-local and sensor evidence alone.
    train_ude::Bool   = false,
    adapt_api::Union{Nothing, Symbol}      = nothing,
    adapt_api_key::Union{Nothing, String}  = nothing,
    adapt_base_url::Union{Nothing, String} = nothing,
    adapt_model::Union{Nothing, String}    = nothing,
    adapt_num_ctx::Union{Nothing, Integer} = nothing,
    adapt_think::Union{Nothing, Bool}      = nothing,
    adapt_input_ranges                     = nothing,
    adapt_min_drift_explained::Real        = 0.25,
    adapt_kwargs::NamedTuple               = NamedTuple(),
    state::AgentState                      = AgentState(),
    train_kwargs...,
)
    result     = nothing
    adaptation = nothing

    verbose && @info "=== AIRMED: $(problem.name) ==="

    # 1. Load measurement data
    verbose && @info "Step 1: Loading measurement data…"
    data_times, data_states = load_data(problem.data_source, problem.tspan, n_points)

    # Extra rows beyond `problem.u0` are sensor channels for the structural
    # analysis; drift detection and training use the dynamic states only.
    n_ude_states = length(problem.u0)
    train_states = size(data_states, 1) > n_ude_states ?
                   data_states[1:n_ude_states, :] : data_states

    # 2. Simulate base model
    verbose && @info "Step 2: Simulating base model…"
    saveat = collect(range(problem.tspan...; length = n_points))
    sol    = simulate(problem; saveat)

    sim_times  = sol.t
    # By symbol in u0-map order, not by row: a DAE solution also contains
    # algebraic unknowns, which must not be compared against measurements.
    sim_states = if problem.u0 isa AbstractVector{<:Pair}
        reduce(vcat, [reshape(Float64.(sol[first(pr)]), 1, :) for pr in problem.u0])
    else
        Array(sol)
    end

    # 3. Compute residuals
    verbose && @info "Step 3: Computing residuals…"
    _, residuals = compute_residuals(sim_times, sim_states, data_times, train_states)
    scalar_res   = vec(mean(residuals; dims = 1))

    verbose && @info "  mean_residual=$(round(mean(scalar_res); sigdigits=4))  " *
                    "max_residual=$(round(maximum(scalar_res); sigdigits=4))"

    # 4. Detect drift
    verbose && @info "Step 4: Detecting drift (method=$(problem.drift_config.method))…"
    drift = detect_drift(scalar_res, problem.drift_config)
    verbose && @info "  drift_detected=$(drift.drift_detected)"

    # 5. Supervision decision
    action = agent_supervise!(state, scalar_res, drift, problem.supervision_config)
    verbose && @info "Step 5: Supervisor action = $action"

    if action === :escalate
        @warn "AIRMED: Escalating to human supervisor. Check supervision report."
    elseif action === :update
        if train_ude
            verbose && @info "Step 6: Training UDE…"
            out_dim = isnothing(nn_output_dim) ? size(train_states, 1) : nn_output_dim
            result  = run_update_workflow(
                problem, drift, base_ode!, nn_input_fn,
                _extract_u0_vec(problem), data_times, train_states;
                nn_input_dim, nn_output_dim = out_dim,
                nn_hidden, nn_depth,
                verbose, train_kwargs...,
            )
            record_update!(state, result)
            verbose && @info "  $(result.message)"
        else
            # `propose_model_adaptation` handles `update_result = nothing`,
            # so skipping training avoids computing an unused UDE.
            verbose && @info "Step 6: Skipped (train_ude = false). Adaptation uses " *
                             "hook-local and sensor evidence only."
        end

        # 7. Structural adaptation. Without a UDE it runs unconditionally;
        # with one it is gated on how much drift that UDE explained.
        ude_ok           = isnothing(result) || result.success
        drift_explained  = isnothing(result) ? nothing : _mean_drift_explained(result)
        if adapt && ude_ok &&
                !isnothing(drift_explained) && drift_explained < adapt_min_drift_explained
            @warn "AIRMED: skipping structural adaptation. The UDE correction " *
                  "explained $(round(100 * drift_explained; digits=1))% of the " *
                  "base-model drift, below the threshold of " *
                  "$(round(100 * Float64(adapt_min_drift_explained); digits=1))%. " *
                  "Consider different network inputs (for example including time), a " *
                  "larger network, more training iterations, or a lower " *
                  "`adapt_min_drift_explained` to run the adaptation regardless."
        elseif adapt && ude_ok
            verbose && @info "Step 7: Proposing structural model adaptation…"
            sup          = problem.supervision_config
            input_ranges = isnothing(adapt_input_ranges) ?
                           (isnothing(result) ? nothing : _auto_input_ranges(result)) :
                           adapt_input_ranges
            adaptation = propose_model_adaptation(
                problem, result;
                data_times, data_states, input_ranges,
                api      = something(adapt_api,      sup.agent_api),
                api_key  = something(adapt_api_key,  sup.agent_api_key),
                base_url = something(adapt_base_url, sup.agent_base_url),
                model    = something(adapt_model,    sup.agent_model),
                # Not `something`: both may be `nothing`, meaning "use the
                # environment variable", which `something` rejects.
                num_ctx  = adapt_num_ctx === nothing ? sup.agent_num_ctx : adapt_num_ctx,
                think    = adapt_think   === nothing ? sup.agent_think   : adapt_think,
                adapt_kwargs...,
            )
            record_adaptation!(state, adaptation)
            verbose && @info "  $(adaptation.message)"
        end
    end

    verbose && println(generate_report(state))
    return state, result, adaptation
end

"""
    _auto_input_ranges(result::UpdateResult) -> Union{Nothing, Vector{Tuple{Float64,Float64}}}

Characterisation ranges derived from the inputs visited along the fitted
trajectory, with a 5% margin per dimension.

# Arguments
- `result`: the training result to read `trajectory_inputs` from.

# Returns
One `(lo, hi)` per input dimension, or `nothing` when the trajectory is
unavailable, in which case the caller skips the characterisation.
"""
function _auto_input_ranges(result::UpdateResult)
    ti = result.trajectory_inputs
    ti isa AbstractMatrix || return nothing
    ranges = Tuple{Float64, Float64}[]
    for d in axes(ti, 1)
        lo, hi = extrema(Float64.(ti[d, :]))
        span   = hi - lo
        margin = span > 1e-12 ? 0.05 * span : max(0.1 * abs(lo), 0.1)
        push!(ranges, (lo - margin, hi + margin))
    end
    return ranges
end

"""
    _mean_drift_explained(result::UpdateResult) -> Union{Float64, Nothing}

Mean of the `:drift_explained` metric across states, skipping NaN entries.

# Arguments
- `result`: the training result to read the metric from.

# Returns
The mean fraction of drift explained, or `nothing` when the metric is
unavailable, in which case the caller does not gate on it.
"""
function _mean_drift_explained(result::UpdateResult)
    ex = get(result.metrics, :drift_explained, nothing)
    ex isa AbstractVector || return nothing
    vals = [Float64(v) for v in ex if !isnan(v)]
    return isempty(vals) ? nothing : mean(vals)
end

function _extract_u0_vec(problem::AIRMEDProblem)
    u0 = problem.u0
    u0 isa AbstractVector{<:Real} && return Float64.(u0)
    # The map order defines the state ordering for `base_ode!`, so its values
    # are taken directly; an ODEProblem u0 would include algebraic unknowns.
    u0 isa AbstractVector{<:Pair} && return Float64.([last(pr) for pr in u0])
    return Float64.(make_ode_problem(problem).u0)
end

end # module
