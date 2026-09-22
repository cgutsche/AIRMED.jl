"""
Agent-in-the-loop supervision: monitors residuals, detects anomalies, escalates.
"""

using Statistics
using Dates

"""
    AgentState

Runtime state of the supervisor, accumulated across iterations.

# Fields
- `n_iterations`, `n_drift_events`, `n_updates`: counters.
- `escalated`: whether an escalation has occurred.
- `residual_history`, `drift_history`: one entry per iteration.
- `update_history`, `adaptation_history`: the recorded results.
- `log`: timestamped log lines.
"""
mutable struct AgentState
    n_iterations::Int
    n_drift_events::Int
    n_updates::Int
    escalated::Bool
    residual_history::Vector{Float64}
    drift_history::Vector{DriftResult}
    update_history::Vector{UpdateResult}
    adaptation_history::Vector{ModelAdaptationResult}
    log::Vector{String}
end

AgentState() = AgentState(0, 0, 0, false, Float64[], DriftResult[], UpdateResult[],
                          ModelAdaptationResult[], String[])

function _log!(state::AgentState, msg::String)
    push!(state.log, "[$( Dates.now() )] $msg")
end

"""
    agent_supervise!(
        state::AgentState,
        residuals::AbstractVector,
        drift::DriftResult,
        config::SupervisionConfig
    ) -> Symbol

Run one supervision step and update `state`.

# Arguments
- `state`: supervision state, mutated in place.
- `residuals`: residuals of the current window.
- `drift`: result of the drift check.
- `config`: thresholds and agent settings.

# Returns
`:continue`, `:update` or `:escalate`.
"""
function agent_supervise!(
    state::AgentState,
    residuals::AbstractVector{<:Real},
    drift::DriftResult,
    config::SupervisionConfig,
)::Symbol
    state.n_iterations += 1
    push!(state.residual_history, mean(residuals))
    push!(state.drift_history, drift)

    if !drift.drift_detected
        _log!(state, "No drift detected. mean_residual=$(round(mean(residuals); sigdigits=4))")
        return :continue
    end

    state.n_drift_events += 1
    _log!(state, "Drift detected (event #$(state.n_drift_events)): " *
                 "method=$(drift.method), max_residual=$(round(drift.max_residual; sigdigits=4))")

    if config.escalate_on_repeated_drift && state.n_drift_events > config.max_auto_updates
        state.escalated = true
        _log!(state, "Escalating to human: repeated drift ($(state.n_drift_events) events).")
        return :escalate
    end

    if drift.max_residual > config.human_threshold && config.ask_human_on_uncertainty
        state.escalated = true
        _log!(state, "Escalating to human: residual $(drift.max_residual) > threshold $(config.human_threshold).")
        return :escalate
    end

    if config.use_agent && config.agent_api !== :none
        decision = _query_llm_agent(state, residuals, drift, config)
        _log!(state, "LLM agent decision: $decision")
        return decision
    end

    return :update
end

"""
    record_update!(state::AgentState, result::UpdateResult)

Record a completed model update.

# Arguments
- `state`: supervision state, mutated in place.
- `result`: the update to record. A successful one also raises `n_updates`.

# Returns
The updated `state`.
"""
function record_update!(state::AgentState, result::UpdateResult)
    push!(state.update_history, result)
    if result.success
        state.n_updates += 1
    end
    _log!(state, "Update recorded: success=$(result.success), " *
                 "final_loss=$(round(result.final_loss; sigdigits=4)), " *
                 "successful_updates=$(state.n_updates)")
end

"""
    record_adaptation!(state::AgentState, result::ModelAdaptationResult)

Record a completed adaptation.

# Arguments
- `state`: supervision state, mutated in place.
- `result`: the adaptation to record.

# Returns
The updated `state`.
"""
function record_adaptation!(state::AgentState, result::ModelAdaptationResult)
    push!(state.adaptation_history, result)
    fit_str = isfinite(result.fit_loss) ?
              "fit_mse=$(round(result.fit_loss; sigdigits=4))" : "fit_mse=n/a"
    _log!(state, "Adaptation recorded: success=$(result.success), " *
                 "api=$(result.api), proposals=$(length(result.proposals)), $fit_str")
end

"""
    generate_report(state::AgentState) -> String

Report the state of the supervision as text.

# Arguments
- `state`: the state to report.

# Returns
The report: iteration counts, drift events, updates, adaptations, the residual
history and the log.
"""
function generate_report(state::AgentState)::String
    io = IOBuffer()
    println(io, "=== AIRMED Supervision Report ===")
    println(io, "Iterations:   $(state.n_iterations)")
    println(io, "Drift events: $(state.n_drift_events)")
    println(io, "Model updates:$(state.n_updates)")
    println(io, "Adaptations:  $(length(state.adaptation_history))")
    println(io, "Escalated:    $(state.escalated)")
    if !isempty(state.residual_history)
        println(io, "Mean residual history: ",
                join(round.(state.residual_history; sigdigits=4), ", "))
    end
    println(io, "--- Log ---")
    for entry in state.log
        println(io, entry)
    end
    return String(take!(io))
end

function _query_llm_agent(
    state::AgentState,
    residuals::AbstractVector,
    drift::DriftResult,
    config::SupervisionConfig,
)::Symbol
    prompt = """
    You are supervising an adaptive digital twin simulation (AIRMED).
    Current status:
    - Drift detected: $(drift.drift_detected)
    - Detection method: $(drift.method)
    - Mean residual: $(round(mean(residuals); sigdigits=4))
    - Max residual: $(round(drift.max_residual; sigdigits=4))
    - Previous drift events: $(state.n_drift_events)
    - Previous model updates: $(state.n_updates)

    Respond with exactly one word — one of: update, escalate, continue.
    Do not include any other text.
    """

    if config.agent_api === :anthropic
        return _supervise_anthropic(prompt, config.agent_api_key;
                                    model = config.agent_model)
    elseif config.agent_api === :openai  ||
           config.agent_api === :ollama  ||
           config.agent_api === :groq    ||
           config.agent_api === :openrouter
        return _supervise_openai(prompt, config.agent_api_key;
                                 base_url = config.agent_base_url,
                                 model    = config.agent_model)
    else
        return :update
    end
end

# Distinct names avoid a dispatch conflict with the model-adaptation _call_anthropic
# (which takes the same argument types but returns a 3-tuple of proposals).

function _supervise_anthropic(prompt::AbstractString, api_key::AbstractString;
                              model::AbstractString = "")::Symbol
    if isempty(api_key)
        @warn "Supervisor: no Anthropic API key, defaulting to :update."
        return :update
    end
    try
        url  = "https://api.anthropic.com/v1/messages"
        hdrs = [
            "x-api-key"         => api_key,
            "anthropic-version" => "2023-06-01",
            "content-type"      => "application/json",
        ]
        model_name = isempty(model) ? "claude-haiku-4-5" : String(model)  # fast/cheap for supervision
        body = JSON3.write(Dict(
            "model"      => model_name,
            "max_tokens" => 16,
            "messages"   => [Dict("role" => "user", "content" => String(prompt))],
        ))
        resp = HTTP.post(url, hdrs, body;
                         connect_timeout = 10, readtimeout = 60, retry = true, retries = 2)
        text = String(JSON3.read(String(resp.body))["content"][1]["text"])
        return _parse_supervision_decision(text)
    catch err
        @warn "Supervisor Anthropic call failed: $(sprint(showerror, err)), defaulting to :escalate."
        return :escalate
    end
end

function _supervise_openai(prompt::AbstractString, api_key::AbstractString;
                           base_url::AbstractString = "",
                           model::AbstractString    = "")::Symbol
    if isempty(api_key) && isempty(base_url)
        @warn "No OpenAI API key supplied for supervisor (and no base_url override), defaulting to :escalate."
        return :escalate
    end
    url        = isempty(base_url) ? "https://api.openai.com/v1/chat/completions" : rstrip(base_url, '/') * "/chat/completions"
    model_name = isempty(model) ? "gpt-4o-mini" : model
    hdrs = isempty(api_key) ? ["content-type" => "application/json"] :
                              ["Authorization" => "Bearer $api_key", "content-type" => "application/json"]
    try
        body = JSON3.write(Dict(
            "model"      => model_name,
            "max_tokens" => 16,
            "messages"   => [Dict("role" => "user", "content" => String(prompt))],
        ))
        resp = HTTP.post(url, hdrs, body;
                         connect_timeout = 10, readtimeout = 60, retry = true, retries = 2)
        text = String(JSON3.read(String(resp.body))["choices"][1]["message"]["content"])
        return _parse_supervision_decision(text)
    catch err
        @warn "Supervisor OpenAI-compatible call failed ($url): $(sprint(showerror, err)), defaulting to :escalate."
        return :escalate
    end
end

function _parse_supervision_decision(text::AbstractString)::Symbol
    t = lowercase(strip(text))
    occursin("escalate", t) && return :escalate
    occursin("update",   t) && return :update
    occursin("continue", t) && return :continue
    @warn "Supervisor: LLM returned unrecognised decision $(repr(text)), defaulting to :escalate."
    return :escalate
end

"""
    explain_for_user(prompt::AbstractString, config::SupervisionConfig) -> String

Turn a technical finding into a short plain-language explanation.

# Arguments
- `prompt`: the finding to explain.
- `config`: supplies the backend through its `agent_*` fields.
- `timeout`: seconds to wait for the response. Default `60`.

# Returns
The explanation, or a bracketed stub when no backend is configured, so the
result can be printed unconditionally.
"""
function explain_for_user(prompt::AbstractString, config::SupervisionConfig;
                          timeout::Real = 60)::String
    if config.agent_api === :anthropic
        return _explain_anthropic(prompt, config.agent_api_key; model = config.agent_model, timeout)
    elseif config.agent_api === :openai || config.agent_api === :ollama ||
           config.agent_api === :groq   || config.agent_api === :openrouter
        return _explain_openai(prompt, config.agent_api_key;
                               base_url = config.agent_base_url, model = config.agent_model, timeout)
    else
        return "[no LLM configured: set agent_api on the SupervisionConfig passed to " *
               "explain_for_user to enable a plain-language explanation]"
    end
end

# Distinct names avoid a dispatch conflict with the supervision and adaptation
# callers, which share the argument types but return a Symbol or a 3-tuple.

function _explain_anthropic(prompt::AbstractString, api_key::AbstractString;
                            model::AbstractString = "", timeout::Real = 60)::String
    if isempty(api_key)
        return "[anthropic stub: no api_key configured]"
    end
    try
        url  = "https://api.anthropic.com/v1/messages"
        hdrs = [
            "x-api-key"         => api_key,
            "anthropic-version" => "2023-06-01",
            "content-type"      => "application/json",
        ]
        model_name = isempty(model) ? "claude-opus-4-8" : String(model)
        body = JSON3.write(Dict(
            "model"      => model_name,
            "max_tokens" => 400,
            "messages"   => [Dict("role" => "user", "content" => String(prompt))],
        ))
        resp = HTTP.post(url, hdrs, body;
                         connect_timeout = 10, readtimeout = round(Int, timeout), retry = true, retries = 2)
        return String(JSON3.read(String(resp.body))["content"][1]["text"])
    catch err
        return "[anthropic error: $(sprint(showerror, err))]"
    end
end

function _explain_openai(prompt::AbstractString, api_key::AbstractString;
                         base_url::AbstractString = "", model::AbstractString = "",
                         timeout::Real = 60)::String
    if isempty(api_key) && isempty(base_url)
        return "[openai stub: no api_key configured]"
    end
    url        = isempty(base_url) ? "https://api.openai.com/v1/chat/completions" : rstrip(base_url, '/') * "/chat/completions"
    model_name = isempty(model) ? "gpt-4o-mini" : model
    hdrs = isempty(api_key) ? ["content-type" => "application/json"] :
                              ["Authorization" => "Bearer $api_key", "content-type" => "application/json"]
    try
        body = JSON3.write(Dict(
            "model"      => model_name,
            "max_tokens" => 400,
            "messages"   => [Dict("role" => "user", "content" => String(prompt))],
        ))
        resp = HTTP.post(url, hdrs, body;
                         connect_timeout = 10, readtimeout = round(Int, timeout), retry = true, retries = 2)
        return String(JSON3.read(String(resp.body))["choices"][1]["message"]["content"])
    catch err
        return "[openai error: $(sprint(showerror, err))]"
    end
end
