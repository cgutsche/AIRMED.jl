"""
Types for the AIRMED framework.
"""

using ModelingToolkit: ODESystem

"""
    ComponentHook

A position in the model at which an unknown component could be inserted, given
by the two ports it would connect.

# Fields
- `name`: identifier of the hook.
- `port_a`, `port_b`: attachment points as `(subsystem, port)`. The same
  subsystem in both means a parallel insertion, two different ones a series
  insertion. Port names follow the caller's component library.
"""
struct ComponentHook
    name::Symbol
    port_a::Tuple{Symbol, Symbol}   # (subsystem_name, port_name)
    port_b::Tuple{Symbol, Symbol}
    description::String
end

"""
    ComponentGuess

A candidate component type the model might be missing. The pipeline decides
which type goes at which hook; parameter values come from fitting.

# Fields
- `component_type`: MTK type name.
- `description`: short description, shown in prompts.
- `factory`: `(name, value) -> ODESystem`, builds the component. All library
  knowledge lives here, so the framework imports no domain package.
- `parameter_name`: the scalar parameter to fit. Default `:p`.
- `default_parameter`: starting value for the fit. Default `1.0`.
- `param_min`, `param_max`: plausible bounds, both positive since fitting runs
  in log-space. A value at a bound triggers a structural retry. Default
  `(1e-12, 1e12)`.
- `connector_names`: the two ports of the produced component. Default `(:p, :n)`.
- `connector_guesses`: initial values for connector variables, needed when MTK
  initialisation is cyclic. Default empty.
- `code_template`: `(name, value) -> String` for the generated declaration line.
  Default `nothing`, which emits the generic `@named` form.
"""
struct ComponentGuess
    component_type::Symbol
    description::String
    factory::Function   # (name::Symbol, param_val::Float64) -> ODESystem
    parameter_name::Symbol
    default_parameter::Float64
    param_min::Float64
    param_max::Float64
    connector_names::Tuple{Symbol, Symbol}
    connector_guesses::Dict{Symbol, Float64}
    code_template::Union{Nothing, Function}
end

# Convenience keyword constructor with sensible generic defaults.
function ComponentGuess(
    component_type::Symbol,
    description::AbstractString,
    factory::Function;
    parameter_name::Symbol            = :p,
    default_parameter::Real           = 1.0,
    param_min::Real                   = 1e-12,
    param_max::Real                   = 1e12,
    connector_names::Tuple{Symbol, Symbol} = (:p, :n),
    connector_guesses::Dict{Symbol, <:Real} = Dict{Symbol, Float64}(),
    code_template::Union{Nothing, Function} = nothing,
)
    param_min > 0 || throw(ArgumentError("param_min must be positive (fitting runs in log-space)"))
    param_max > param_min || throw(ArgumentError("param_max must exceed param_min"))
    ComponentGuess(component_type, String(description), factory,
                   parameter_name, Float64(default_parameter),
                   Float64(param_min), Float64(param_max),
                   connector_names,
                   Dict{Symbol, Float64}(k => Float64(v) for (k, v) in connector_guesses),
                   code_template)
end

"""
    ComponentProposal

One concrete insertion: which component type connects which two ports. Unlike
`ComponentGuess`, this is the outcome of an analysis.

# Fields
- `name`: variable name used in the generated code.
- `component_type`: must match a declared `ComponentGuess`.
- `port_a`, `port_b`: attachment points, mapped to the two `connector_names`.
- `parameters`: parameter values by symbol.
- `rationale`: justification given for the proposal.
"""
struct ComponentProposal
    name::Symbol
    component_type::Symbol
    port_a::Tuple{Symbol, Symbol}
    port_b::Tuple{Symbol, Symbol}
    parameters::Dict{Symbol, Float64}
    rationale::String
end

"""
    DriftDetectionMethod

Supertype for the method-specific parameters of a drift detector, one instance
of which is held by `DriftConfig.method`. A new algorithm needs a subtype and a
`_detect` method in `validation.jl`. The baseline settings shared by all methods
stay in `DriftConfig`.
"""
abstract type DriftDetectionMethod end

"""
    CUSUM(; k=0.5, h=5.0)

Cumulative-sum drift detection. See `_cusum_detect`.

# Fields
- `k`: slack in baseline standard deviations, below which a deviation is not
  accumulated. Default `0.5`.
- `h`: bound on the accumulated evidence that signals drift. Default `5.0`.
"""
@kwdef struct CUSUM <: DriftDetectionMethod
    k::Float64 = 0.5
    h::Float64 = 5.0
end

"""
    EWMA(; lambda=0.2, L=3.0)

Exponentially-weighted moving-average drift detection. See `_ewma_detect`.

# Fields
- `lambda`: smoothing factor. Default `0.2`.
- `L`: control limit, as a multiple of the adjusted baseline standard
  deviation. Default `3.0`.
"""
@kwdef struct EWMA <: DriftDetectionMethod
    lambda::Float64 = 0.2
    L::Float64      = 3.0
end

"""
    SimpleThreshold(; threshold=0.1)

Flag drift as soon as a residual exceeds the threshold, without baseline
estimation or accumulated evidence. See `_threshold_detect`.

# Fields
- `threshold`: the absolute bound. Default `0.1`.
"""
@kwdef struct SimpleThreshold <: DriftDetectionMethod
    threshold::Float64 = 0.1
end

"""
    DriftConfig

Configuration for drift detection.

# Fields
- `method`: the algorithm and its own parameters, e.g. `CUSUM(k=1.0, h=15.0)`.
  Default `CUSUM()`.
- `window_size`: samples per detection window. Default `50`.
- `min_samples`: samples used for the baseline, below which no detection runs.
  Default `10`.
- `min_sigma`: lower bound on the baseline standard deviation, which prevents
  spurious detections at residuals near machine precision. Default `1e-8`.
"""
@kwdef struct DriftConfig
    method::DriftDetectionMethod = CUSUM()
    window_size::Int      = 50
    min_samples::Int      = 10
    # Lower bound on the baseline standard deviation used by CUSUM and EWMA,
    # preventing spurious detections at residuals near machine precision.
    min_sigma::Float64    = 1e-8
end

"""
    SupervisionConfig

Configuration of the supervisor `agent_supervise!` and of every LLM call the
framework makes on its behalf.

The escalation conditions are checked first; the agent is consulted only if
neither fires, and cannot override them.

# Fields
- `use_agent`: ask the LLM for the decision when no escalation condition fired.
  Otherwise that case becomes `:update`. Default `false`.
- `agent_api`: backend, one of `:anthropic`, `:openai`, `:ollama`, `:groq`,
  `:openrouter`, `:none`. `:none` calls no LLM anywhere. Default `:none`.
- `agent_api_key`: key for that backend; unused for `:none` and local servers.
- `agent_base_url`: endpoint override, required for `:ollama`.
- `agent_model`: model name; empty selects a per-call default.
- `escalate_on_repeated_drift`: escalate once the drift events exceed
  `max_auto_updates`, capping unattended updates. Default `true`.
- `max_auto_updates`: that limit. Default `3`.
- `human_threshold`: residual above which one drift event escalates, if
  `ask_human_on_uncertainty` is set. Default `0.5`.
- `ask_human_on_uncertainty`: enables the `human_threshold` check. Default `true`.
- `agent_num_ctx`, `agent_think`: context size and reasoning pass for
  `:ollama_native` adaptation calls. `nothing` uses `AIRMED_OLLAMA_NUM_CTX` and
  `AIRMED_OLLAMA_THINK`. Reasoning shares the token allowance of the answer.
"""
@kwdef struct SupervisionConfig
    use_agent::Bool                    = false
    agent_api::Symbol                  = :none
    agent_api_key::String              = ""
    agent_base_url::String             = ""   # override for Ollama, Groq, OpenRouter, etc.
    agent_model::String                = ""   # override default model name
    escalate_on_repeated_drift::Bool   = true
    max_auto_updates::Int              = 3
    human_threshold::Float64           = 0.5
    ask_human_on_uncertainty::Bool     = true
    # :ollama_native only; `nothing` falls back to AIRMED_OLLAMA_NUM_CTX and
    # AIRMED_OLLAMA_THINK.
    agent_num_ctx::Union{Nothing, Int} = nothing
    agent_think::Union{Nothing, Bool}  = nothing
end

abstract type DataSource end

"""
    CSVDataSource

Data source backed by a CSV file.

# Fields
- `filepath`: the file to read.
- `time_column`: column holding the time vector.
- `state_columns`: columns holding the observed states, in row order.
- `input_columns`: optional columns holding inputs.
"""
struct CSVDataSource <: DataSource
    filepath::String
    time_column::Symbol
    state_columns::Vector{Symbol}
    input_columns::Vector{Symbol}

    CSVDataSource(filepath, time_col, state_cols, input_cols=Symbol[]) =
        new(filepath, time_col, state_cols, input_cols)
end

"""
    FunctionDataSource

Data source from a callable, e.g. for synthetic or test data.

# Fields
- `generator`: called as `(tspan, n_points) -> (times, states)`.
"""
struct FunctionDataSource <: DataSource
    generator::Function
end

"""
    DriftResult

Result of a drift check.

# Fields
- `drift_detected`: the verdict.
- `drift_index`: first index at which the bound was crossed, or `nothing`.
- `mean_residual`, `max_residual`: statistics of the series checked.
- `method`: which detector produced the result.
- `details`: statistics of that method, e.g. the CUSUM series.
"""
struct DriftResult
    drift_detected::Bool
    drift_index::Union{Int, Nothing}
    mean_residual::Float64
    max_residual::Float64
    method::Symbol
    details::Dict{Symbol, Any}
end

"""
    UpdateResult

Result of a model update attempt.

# Fields
- `success`, `final_loss`, `n_iterations`, `message`: outcome of the training.
- `nn`, `trained_nn_params`, `trained_nn_state`: the trained network.
- `sym_nn`, `sym_theta`, `fitted_ode_prob`: set only on the symbolic path of
  `train_symbolic_ude`, otherwise `nothing`.
- `trajectory_inputs`: network inputs along the fitted trajectory, so later
  sampling stays on the visited manifold instead of a uniform grid.
- `metrics`: diagnostics such as `:per_state_rmse_base`, `:per_state_rmse_ude`
  and `:drift_explained`, the fraction of base-model residual removed.
"""
struct UpdateResult
    success::Bool
    final_loss::Float64
    n_iterations::Int
    nn               # Lux network architecture (needed for printing and inference)
    trained_nn_params
    trained_nn_state
    message::String
    sym_nn::Any          # symbolic NN from @SymbolicNeuralNetwork (nothing if Lux path)
    sym_theta::Any       # θ parameter vector from @SymbolicNeuralNetwork (nothing if Lux path)
    fitted_ode_prob::Any # fitted ODEProblem after symbolic training (nothing if Lux path)
    trajectory_inputs::Any    # (n_inputs × n_times) NN inputs along fitted trajectory, or nothing
    metrics::Dict{Symbol, Any}
end

# Backward-compatible 7-argument constructor for the plain-Lux training path.
function UpdateResult(success, final_loss, n_iterations, nn, trained_nn_params,
                      trained_nn_state, message)
    UpdateResult(success, final_loss, n_iterations, nn, trained_nn_params,
                 trained_nn_state, message, nothing, nothing, nothing,
                 nothing, Dict{Symbol, Any}())
end

# Backward-compatible 10-argument constructor (symbolic-UDE path).
function UpdateResult(success, final_loss, n_iterations, nn, trained_nn_params,
                      trained_nn_state, message, sym_nn, sym_theta, fitted_ode_prob)
    UpdateResult(success, final_loss, n_iterations, nn, trained_nn_params,
                 trained_nn_state, message, sym_nn, sym_theta, fitted_ode_prob,
                 nothing, Dict{Symbol, Any}())
end

"""
    AdaptableParam

An existing model parameter that may be re-fitted instead of inserting a new
component. `propose_model_adaptation` tries each one individually before
building a prompt or searching the hooks.

# Fields
- `name`: identifier of the parameter.
- `param`: the symbolic parameter, which must appear in `problem.p0`.
- `default_value`: nominal value, the starting point of the fit.
- `param_min`, `param_max`: plausible bounds, both positive since fitting runs
  in log-space. Default `(1e-12, 1e12)`.
- `description`: short description for reports and logs.
"""
struct AdaptableParam
    name::Symbol
    param::Any
    default_value::Float64
    param_min::Float64
    param_max::Float64
    description::String
end

function AdaptableParam(
    name::Symbol,
    param,
    default_value::Real,
    description::AbstractString = "";
    param_min::Real = 1e-12,
    param_max::Real = 1e12,
)
    param_min > 0 || throw(ArgumentError("param_min must be positive (fitting runs in log-space)"))
    param_max > param_min || throw(ArgumentError("param_max must exceed param_min"))
    AdaptableParam(name, param, Float64(default_value),
                   Float64(param_min), Float64(param_max), String(description))
end

"""
    ParameterAdjustment

A re-fit of an existing parameter. The structure is unchanged and only the
value is corrected.

# Fields
- `name`, `param`: taken from the originating `AdaptableParam`.
- `old_value`: the value in `problem.p0` before the adjustment.
- `new_value`: the fitted value.
- `rationale`: justification for the adjustment.
"""
struct ParameterAdjustment
    name::Symbol
    param::Any
    old_value::Float64
    new_value::Float64
    rationale::String
end

"""
    LLMExchange

One prompt and response pair. `propose_model_adaptation` may call the LLM up
to `max_retries` times, and keeps every exchange here, not only the accepted one.

# Fields
- `attempt`: attempt index, starting at 1.
- `prompt`, `raw_response`: what was sent and received, verbatim.
- `n_parsed`: number of proposals parsed from the response.
- `note`: outcome, e.g. accepted, retried or discarded.
"""
struct LLMExchange
    attempt::Int
    prompt::String
    raw_response::String
    n_parsed::Int
    note::String
end

"""
    ModelAdaptationResult

Result of `propose_model_adaptation`.

# Fields
- `proposals`: components to insert; empty for a parameter-only fix.
- `parameter_adjustments`: parameters re-fitted in place; empty for a
  structural fix.
- `generated_code`: MTK source for the adapted system, regenerated from the
  final proposals rather than taken from the response.
- `adapted_model`: the built `ODESystem`, or `nothing`.
- `prompt`, `raw_response`: those of the accepted attempt.
- `conversation`: every attempt in order, including discarded ones.
- `api`: the backend used.
- `success`: whether the pipeline produced a usable result.
- `message`: status summary, including the structure-quality verdict.
- `fit_loss`: MSE after fitting, `NaN` if fitting was skipped or failed.
- `escalation`: explanation for a human, empty on ordinary paths.
"""
struct ModelAdaptationResult
    proposals::Vector{ComponentProposal}
    parameter_adjustments::Vector{ParameterAdjustment}
    generated_code::String
    adapted_model::Union{ODESystem, Nothing}
    prompt::String
    raw_response::String
    api::Symbol
    success::Bool
    message::String
    fit_loss::Float64
    conversation::Vector{LLMExchange}
    # Account of why no adaptation could be made, for a human. Empty unless
    # `explain_on_failure` is set and the LLM produced nothing usable.
    escalation::String
end

# 11-argument constructor: `escalation` defaults to empty, so the field is set
# only on the escalation path.
function ModelAdaptationResult(proposals, parameter_adjustments, generated_code,
                                adapted_model, prompt, raw_response, api, success,
                                message, fit_loss, conversation)
    ModelAdaptationResult(proposals, parameter_adjustments, generated_code, adapted_model,
                          prompt, raw_response, api, success, message, fit_loss,
                          conversation, "")
end

# 10-argument constructor: `conversation` defaults to empty for callers that
# make no LLM call.
function ModelAdaptationResult(proposals, parameter_adjustments, generated_code,
                                adapted_model, prompt, raw_response, api, success,
                                message, fit_loss)
    ModelAdaptationResult(proposals, parameter_adjustments, generated_code, adapted_model,
                          prompt, raw_response, api, success, message, fit_loss,
                          LLMExchange[])
end

# 9-argument constructor: `parameter_adjustments` defaults to empty, for
# callers that produce structural proposals only.
function ModelAdaptationResult(proposals, generated_code, adapted_model,
                                prompt, raw_response, api, success, message, fit_loss)
    ModelAdaptationResult(proposals, ParameterAdjustment[], generated_code, adapted_model,
                          prompt, raw_response, api, success, message, fit_loss,
                          LLMExchange[])
end

# Backward-compatible 8-argument constructor (fit_loss defaults to NaN).
function ModelAdaptationResult(proposals, generated_code, adapted_model,
                                prompt, raw_response, api, success, message)
    ModelAdaptationResult(proposals, generated_code, adapted_model,
                          prompt, raw_response, api, success, message, NaN)
end

"""
    HookResidualSpec

The balance law that holds at one `ComponentHook` if the model is complete. If
it does not close, the imbalance is the through- or across-variable of the
missing element, measured without simulation. Supplied by the developer, since
deriving it would require the connection-set internals of MTK.

# Fields
- `hook`: name of the hook this describes.
- `kind`: `:parallel` for a flow response from a node balance, `:series` for a
  potential response from a loop balance.
- `drive`: the observable read as the drive.
- `response_terms`: `observable => coefficient` pairs summed into the response,
  with the sign convention of the model.
- `description`: short description, shown in prompts.

# Example

    HookResidualSpec(:load_parallel, :parallel, r_load.v,
                     [r_grid.i => 1.0, r_pv.i => 1.0, bat.i => -1.0,
                      r_load.i => -1.0, r_const.i => -1.0],
                     "extra path parallel to the consumers")
"""
struct HookResidualSpec
    hook::Symbol
    kind::Symbol
    drive::Any
    response_terms::Vector{Pair{Any, Float64}}
    description::String
end

function HookResidualSpec(hook::Symbol, kind::Symbol, drive, response_terms,
                          description::AbstractString = "")
    kind in (:parallel, :series) ||
        throw(ArgumentError("HookResidualSpec kind must be :parallel or :series, got $kind"))
    return HookResidualSpec(hook, kind, drive,
                            Pair{Any, Float64}[k => Float64(v) for (k, v) in response_terms],
                            String(description))
end

"""
    AIRMEDProblem

The problem definition: the model, its data source, and the detection and
supervision settings. Built with the keyword constructor, which simplifies the
model and validates the hook declarations.

# Fields
- `name`, `model`, `simplified_model`: identifier and the MTK model, before and
  after `structural_simplify`.
- `tspan`, `u0`, `p0`: time span, initial conditions and parameters.
- `data_source`: where the measurements come from.
- `drift_config`, `supervision_config`: detection and supervision settings.
- `component_hooks`, `component_guesses`: where a component may be inserted and
  which types are candidates.
- `adaptable_params`, `optimizable_params`: parameters that may be re-fitted or
  calibrated.
- `observable_states`, `observation_noise`: the sensors and their 1 sigma noise.
- `hook_residuals`: the balance law per hook, for the hook-local step.
- `model_preamble`: `using` lines written into the generated code.
- `port_aliases`, `component_library_label`: naming of the caller's library.
- `adaptation_guesses`: initial values for an ambiguous initialisation.
"""
struct AIRMEDProblem
    name::String
    model::ODESystem
    simplified_model::ODESystem
    component_hooks::Vector{ComponentHook}
    component_guesses::Vector{ComponentGuess}
    # Existing parameters that may be re-fitted directly, tried before the
    # structural search. See `AdaptableParam`.
    adaptable_params::Vector{AdaptableParam}
    optimizable_params::Vector{Symbol}
    data_source::DataSource
    drift_config::DriftConfig
    supervision_config::SupervisionConfig
    tspan::Tuple{Float64, Float64}
    u0
    p0
    # `using` / `import` lines written into generated adaptation code.
    # Provided by the developer; AIRMED imports no domain libraries itself.
    model_preamble::Vector{String}
    # State variables with sensors, as rows of `data_states`; only these are
    # compared during fitting. If empty, the first n_data rows are used.
    observable_states::Vector
    # Optional aliases mapping port names emitted by the LLM to those of the
    # component library. Empty by default, as no domain is assumed.
    port_aliases::Dict{String, String}
    # Label for the component library shown in LLM prompts, e.g.
    # "ModelingToolkitStandardLibrary.Electrical". Informational only.
    component_library_label::String
    # Initial values for existing-component variables whose initialisation
    # becomes ambiguous once a new component is wired in. Keys as in u0.
    adaptation_guesses::Dict{Any, Any}
    # Per-hook balance laws for the hook-local measurement (see
    # `HookResidualSpec`). Empty by default, which skips that step.
    hook_residuals::Vector{HookResidualSpec}
    # 1 sigma noise per row of `data_states`, in that row's units. Without it
    # no signal-to-noise statement is possible and the hook step is skipped.
    observation_noise::Vector{Float64}
end

function AIRMEDProblem(;
    name::String,
    model::ODESystem,
    component_hooks::Vector{ComponentHook}            = ComponentHook[],
    component_guesses::Vector{ComponentGuess} = ComponentGuess[],
    adaptable_params::Vector{AdaptableParam} = AdaptableParam[],
    optimizable_params::Vector{Symbol}  = Symbol[],
    data_source::DataSource,
    drift_config::DriftConfig           = DriftConfig(),
    supervision_config::SupervisionConfig = SupervisionConfig(),
    tspan::Tuple{<:Real, <:Real},
    u0,
    p0,
    model_preamble::Vector{String}      = String[],
    observable_states::Vector           = [],
    port_aliases::Dict{<:AbstractString, <:AbstractString} = Dict{String, String}(),
    component_library_label::AbstractString = "caller-supplied component library",
    adaptation_guesses::Dict            = Dict{Any, Any}(),
    hook_residuals::Vector{HookResidualSpec} = HookResidualSpec[],
    observation_noise::AbstractVector{<:Real} = Float64[],
)
    tspan_f64  = (Float64(tspan[1]), Float64(tspan[2]))
    simplified = structural_simplify(model)
    aliases    = Dict{String, String}(String(k) => String(v) for (k, v) in port_aliases)

    if !isempty(hook_residuals)
        isempty(observable_states) &&
            throw(ArgumentError("hook_residuals declared but observable_states is empty; " *
                                "the residual terms are looked up by row in data_states"))
        isempty(observation_noise) &&
            @warn "hook_residuals declared but observation_noise is empty; the hook-local " *
                  "characterisation is skipped, since no noise floor is available."
        !isempty(observation_noise) && length(observation_noise) != length(observable_states) &&
            throw(ArgumentError("observation_noise has $(length(observation_noise)) entries but " *
                                "observable_states has $(length(observable_states))"))
        hooknames = Set(h.name for h in component_hooks)
        for s in hook_residuals
            s.hook in hooknames ||
                throw(ArgumentError("HookResidualSpec references unknown hook $(s.hook); " *
                                    "declared hooks: $(join(sort(collect(hooknames)), ", "))"))
        end
    end

    return AIRMEDProblem(name, model, simplified, component_hooks, component_guesses,
                        adaptable_params, optimizable_params, data_source, drift_config,
                        supervision_config, tspan_f64, u0, p0,
                        model_preamble, observable_states,
                        aliases, String(component_library_label),
                        Dict{Any, Any}(adaptation_guesses),
                        hook_residuals, Float64.(observation_noise))
end
