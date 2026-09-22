# --- Rebuilding the adapted twin --------------------------------------------
# Converts a diagnosis into a runnable problem; see the documentation.

"""
    build_adapted_system(problem, proposals; init_vals = nothing)
        -> (sys | nothing, guesses, msg, comps | nothing)

Rebuild the model of `problem` with `proposals` inserted, returning the
unsimplified `ODESystem`, since `AIRMEDProblem` simplifies it itself and the
raw system still allows symbolic indexing such as `sol[load.i]`.

# Arguments
- `problem`: the problem to rebuild from.
- `proposals`: components to insert.
- `init_vals`: initial parameter values. `nothing` uses each proposal's fitted
  value, falling back to `ComponentGuess.default_parameter`.

# Returns
`(sys, guesses, msg, comps)`, where `comps` holds the created components in
proposal order and `sys` is `nothing` on failure.
"""
function build_adapted_system(problem::AIRMEDProblem,
                              proposals::Vector{ComponentProposal};
                              init_vals::Union{Nothing, Vector{Float64}} = nothing)
    isempty(proposals) && return nothing, Dict{Any,Any}(), "no proposals to insert", nothing
    vals = isnothing(init_vals) ? _proposal_init_vals(problem, proposals) : init_vals
    length(vals) == length(proposals) ||
        return nothing, Dict{Any,Any}(),
               "init_vals has $(length(vals)) entries but there are $(length(proposals)) proposals",
               nothing
    raw, _, msg, guesses, comps = _assemble_adapted_system(proposals, vals, problem)
    return raw, guesses, msg, comps
end

# Fitted parameter value per proposal, falling back to the ComponentGuess
# default when the named parameter is not accepted by the factory.
function _proposal_init_vals(problem::AIRMEDProblem,
                             proposals::Vector{ComponentProposal})::Vector{Float64}
    map(proposals) do p
        gi = findfirst(g -> g.component_type == p.component_type, problem.component_guesses)
        isnothing(gi) && return 0.0
        g = problem.component_guesses[gi]
        Float64(get(p.parameters, g.parameter_name, g.default_parameter))
    end
end

"""
    rebuild_problem(problem; kwargs...) -> AIRMEDProblem

Copy `problem`, overriding only the fields named in `kwargs`.

# Arguments
- `problem`: the problem to copy.

# Keywords
Any field of `AIRMEDProblem`, e.g. `model`, `data_source` or
`observable_states`. Fields not named are carried over, so replacing the model
retains the solver window, the initial conditions and the drift settings.

# Returns
The copied problem.
"""
function rebuild_problem(problem::AIRMEDProblem; kwargs...)
    base = (
        name                    = problem.name,
        model                   = problem.model,
        component_hooks         = problem.component_hooks,
        component_guesses       = problem.component_guesses,
        adaptable_params        = problem.adaptable_params,
        optimizable_params      = problem.optimizable_params,
        data_source             = problem.data_source,
        drift_config            = problem.drift_config,
        supervision_config      = problem.supervision_config,
        tspan                   = problem.tspan,
        u0                      = problem.u0,
        p0                      = problem.p0,
        model_preamble          = problem.model_preamble,
        observable_states       = problem.observable_states,
        port_aliases            = problem.port_aliases,
        component_library_label = problem.component_library_label,
        adaptation_guesses      = problem.adaptation_guesses,
        hook_residuals          = problem.hook_residuals,
        observation_noise       = problem.observation_noise,
    )
    return AIRMEDProblem(; merge(base, NamedTuple(kwargs))...)
end

# The rebuilt twin is for simulation, not further adaptation. The residuals
# are cleared with the hooks, which the constructor requires.
const _TWIN_SEARCH_SPACE_CLEARED = (
    component_hooks   = ComponentHook[],
    adaptable_params  = AdaptableParam[],
    hook_residuals    = HookResidualSpec[],
    observation_noise = Float64[],
)

"""
    build_adapted_problem(problem, proposals; kwargs...) -> (prob | nothing, msg, comps | nothing)

The structural fix as a runnable problem: the model with `proposals`
inserted, every other setting inherited.

# Keywords
- `name`: name of the rebuilt problem.
- `data_source`: replay source; `nothing` keeps the original.
- `observable_states`: narrow this to judge the twin on fewer channels.
- `init_vals`: initial parameter values, as in `build_adapted_system`.

# Returns
`(prob, msg, comps)`, or `(nothing, msg, nothing)` if the system cannot be
built.
"""
function build_adapted_problem(problem::AIRMEDProblem,
                               proposals::Vector{ComponentProposal};
                               name::AbstractString = "adapted twin",
                               data_source::Union{Nothing, DataSource} = nothing,
                               observable_states::Union{Nothing, Vector} = nothing,
                               init_vals::Union{Nothing, Vector{Float64}} = nothing)
    sys, guesses, msg, comps = build_adapted_system(problem, proposals; init_vals)
    isnothing(sys) && return nothing, msg, nothing
    prob = rebuild_problem(problem;
        name               = String(name),
        model              = sys,
        data_source        = isnothing(data_source) ? problem.data_source : data_source,
        observable_states  = isnothing(observable_states) ? problem.observable_states :
                                                            observable_states,
        adaptation_guesses = guesses,
        _TWIN_SEARCH_SPACE_CLEARED...)
    return prob, msg, comps
end

"""
    build_parameter_adjusted_problem(problem, adjustments; kwargs...) -> (prob, msg)

The parameter-only fix as a runnable problem: identical topology, with each
adjusted value replaced in `p0`.

# Arguments
- `problem`: the problem to rebuild.
- `adjustments`: the re-fitted parameters to apply.

# Keywords
- `name`, `data_source`, `observable_states`: as in `build_adapted_problem`.

# Returns
`(prob, msg)`. Adjustments naming a parameter absent from `p0` are skipped and
reported in `msg`.
"""
function build_parameter_adjusted_problem(problem::AIRMEDProblem,
                                          adjustments;
                                          name::AbstractString = "adapted twin (parameter fix)",
                                          data_source::Union{Nothing, DataSource} = nothing,
                                          observable_states::Union{Nothing, Vector} = nothing)
    p0_adjusted = copy(problem.p0)
    applied, missed = 0, String[]
    for adj in adjustments
        idx = findfirst(pr -> isequal(first(pr), adj.param), p0_adjusted)
        if isnothing(idx)
            push!(missed, string(adj.param))
        else
            p0_adjusted[idx] = adj.param => adj.new_value
            applied += 1
        end
    end
    msg = "parameter-adjusted model built ($(applied) parameter(s))" *
          (isempty(missed) ? "" : "; not in p0, skipped: $(join(missed, ", "))")
    prob = rebuild_problem(problem;
        name              = String(name),
        p0                = p0_adjusted,
        data_source       = isnothing(data_source) ? problem.data_source : data_source,
        observable_states = isnothing(observable_states) ? problem.observable_states :
                                                           observable_states,
        _TWIN_SEARCH_SPACE_CLEARED...)
    return prob, msg
end

"""
    build_fix_problem(problem, adapt_result; kwargs...) -> (prob | nothing, msg, comps | nothing)

Turn either form of a `ModelAdaptationResult` into a runnable problem: a
parameter fix or a structural fix, the former taking precedence when both are
present.

# Returns
`(prob, msg, comps)`. `comps` is `nothing` for a parameter fix, since no
component is created.
"""
function build_fix_problem(problem::AIRMEDProblem, adapt_result;
                           name::AbstractString = "adapted twin",
                           data_source::Union{Nothing, DataSource} = nothing,
                           observable_states::Union{Nothing, Vector} = nothing)
    if !isempty(adapt_result.parameter_adjustments)
        prob, msg = build_parameter_adjusted_problem(
            problem, adapt_result.parameter_adjustments;
            name, data_source, observable_states)
        return prob, msg, nothing
    end
    isempty(adapt_result.proposals) &&
        return nothing, "adaptation carried neither proposals nor parameter adjustments", nothing
    return build_adapted_problem(problem, adapt_result.proposals;
                                 name, data_source, observable_states)
end
