
"""
    propose_model_adaptation(problem, update_result = nothing; kwargs...)
        -> ModelAdaptationResult

Diagnose a drift as a re-fitted existing parameter or as a missing component.
Always returns a `ModelAdaptationResult`; on failure `success = false` and
`message` says why.

Each `problem.adaptable_params` entry is re-fitted first, before any prompt or
hook search. If one explains the drift it is returned as a
`ParameterAdjustment`. Otherwise the structural search runs, and
`_winnow_proposals` keeps the smallest subset of a response that still explains
the drift.

# Arguments
- `problem`: the problem whose model is adapted.
- `update_result`: an `UpdateResult` from `train_ude`, or `nothing` to run
  without a UDE.

# Keywords
- `data_times`, `data_states`: the measurements, required for fitting.
- `api`, `api_key`, `base_url`, `model`: the LLM backend. See the API contracts
  for the supported values.
- `num_ctx`, `think`: `:ollama_native` only; `nothing` uses the environment.
- `llm_timeout`, `llm_log_file`: seconds per call, and a file to log prompts to.
- `fit_params`: fit each proposed parameter to the data. Default `true`.
- `fit_iters`, `fit_lr`, `optimization_algorithm`: fitting settings. LBFGS is
  tried first, so `fit_lr` applies to the Adam fallback.
- `holdout_fraction`: tail withheld from fitting, used for validation.
- `complexity_penalty`: ranks attempts by
  `fit_loss * (1 + complexity_penalty * n_components)`; `0` ranks by raw loss.
- `max_retries`: re-proposal attempts after the first. Default `3`.
- `hook_local_analysis`, `hook_summary`, `hook_terminal_inputs`: control the
  hook-local characterisation.
- `restrict_hooks_by_snr`, `hook_snr_min`: offer only hooks above an SNR.
- `include_component_equations`, `include_ude_sections`: prompt content.
- `escalate_on_inadequate_structure`, `explain_on_failure`: reject an
  inadequate result instead of returning it.
- `build_model`: build the generated code into an `ODESystem`.
- `input_ranges`, `n_samples`: grid for the UDE characterisation.

# Returns
A `ModelAdaptationResult`, also on failure, where `success` is `false` and
`message` says why.
"""
function propose_model_adaptation(
    problem::AIRMEDProblem,
    # Optional: a UDE is only described in the prompt, never used by the
    # adequacy test, so `nothing` runs the pipeline without one.
    update_result::Union{UpdateResult, Nothing} = nothing;
    data_times::Union{Nothing, AbstractVector}  = nothing,
    data_states::Union{Nothing, AbstractMatrix} = nothing,
    input_ranges::Union{Nothing, Vector{<:Tuple{<:Real, <:Real}}} = nothing,
    n_samples::Int  = 50,
    api::Symbol     = :none,
    api_key::String = "",
    base_url::String = "",
    model::String    = "",
    # `:ollama_native` only; `nothing` falls back to AIRMED_OLLAMA_NUM_CTX
    # and AIRMED_OLLAMA_THINK.
    num_ctx::Union{Nothing, Integer} = nothing,
    think::Union{Nothing, Bool}      = nothing,
    llm_log_file::Union{Nothing, String} = nothing,
    # HTTP read timeout per call. Local backends on CPU need considerably
    # more than the 180 s that suffice for cloud APIs.
    llm_timeout::Real = 180,
    build_model::Bool = false,
    fit_params::Bool  = true,
    optimization_algorithm = OptimizationOptimJL.LBFGS(),
    fit_iters::Int    = 300,
    fit_lr::Real      = 5e-2,
    # max_retries: re-proposal attempts after the first; 0 disables retries.
    # retry_ratio_threshold: inert, retained for API compatibility.
    max_retries::Int            = 3,
    retry_ratio_threshold::Real = 10.0,
    # complexity_penalty ranks attempts by fit_loss * (1 + penalty * n_comp);
    # holdout_fraction withholds a tail that the reported loss still covers.
    complexity_penalty::Real    = 0.05,
    # Offer only hooks whose balance-law SNR exceeds `hook_snr_min`. With one
    # surviving hook the localisation is given away.
    restrict_hooks_by_snr::Bool = false,
    hook_snr_min::Real          = 3.0,
    holdout_fraction::Real      = 0.25,
    # With `false` the balance-law residual and its SNR are kept but not the
    # analysis, so the prompt locates the residual without describing it.
    hook_local_analysis::Bool         = true,
    # Pre-computed hook-local section. Multithreaded BLAS makes repeated
    # calls differ in the last digit, so pass it to keep prompts identical.
    hook_summary::Union{Nothing, AbstractString} = nothing,
    # List each offered type with its constitutive equations rather than its
    # name alone. Changes the prompt, so false by default.
    include_component_equations::Bool = false,
    # Fit the residual in the element's own terminal quantities instead of
    # every channel. See "Choice of candidate inputs" in the documentation.
    hook_terminal_inputs::Bool = false,
    # Reject a best attempt still above 5 sigma once the budget is spent,
    # instead of returning it. See the escalation block after the loop.
    escalate_on_inadequate_structure::Bool = false,
    # Omit every global-UDE section from the prompt, for pipelines that
    # diagnose from the hook-local step.
    include_ude_sections::Bool  = true,
    # With `true`, an unusable response is explained in `escalation` instead
    # of being replaced by a demo-mode enumeration.
    explain_on_failure::Bool    = false,
)
    # 1. Existing-parameter pre-pass: re-fit each `adaptable_params` entry
    # individually, before any characterisation, prompt or hook search.
    if !isempty(problem.adaptable_params) && !isnothing(data_times) && !isnothing(data_states)
        best_adj = _fit_existing_parameters(problem, data_times, data_states;
                                            max_iters = fit_iters, lr = fit_lr, holdout_fraction, optimization_algorithm = optimization_algorithm)
        if !isnothing(best_adj) && !best_adj.drift_result.drift_detected && !best_adj.at_bound
            plateau_note = best_adj.from_plateau ?
                " Fitted against the settled TAIL of the window only (a plateau was " *
                "detected after an earlier transient) — this value explains the CURRENT, " *
                "settled state, not necessarily the window's earlier samples." : ""
            adj = ParameterAdjustment(best_adj.name, best_adj.param,
                best_adj.old_value, best_adj.new_value,
                "Re-fitted existing parameter $(best_adj.name): " *
                "$(round(best_adj.old_value; sigdigits=4)) → $(round(best_adj.new_value; sigdigits=4)) " *
                "fully explains the drift — no new component needed.$plateau_note")
            @info "Adaptation: existing-parameter fit ($(best_adj.name) = " *
                  "$(round(best_adj.new_value; sigdigits=4)), was $(round(best_adj.old_value; sigdigits=4))) " *
                  "fully explains the drift (fit_loss=$(round(best_adj.fit_loss; sigdigits=4))" *
                  (best_adj.from_plateau ? ", fit against detected plateau" : "") * ") — " *
                  "accepting without structural search."
            gen_code = """
            # Auto-generated by AIRMED.propose_model_adaptation
            #
            # PARAMETER ADJUSTMENT — no structural change. Update this existing
            # parameter's value in your model script (e.g. the `@named`
            # declaration's keyword, or the corresponding `p0` entry):
            #
            #   $(adj.name) : $(round(adj.old_value; sigdigits=6)) → $(round(adj.new_value; sigdigits=6))
            """
            return ModelAdaptationResult(
                ComponentProposal[],
                [adj],
                gen_code,
                nothing,
                "",
                "[parameter-only pre-pass — no LLM call needed]",
                :none,
                true,
                "Model adaptation resolved via a PARAMETER ADJUSTMENT " *
                "($(adj.name): $(round(adj.old_value; sigdigits=4)) → " *
                "$(round(adj.new_value; sigdigits=4))) — no new component needed.",
                best_adj.fit_loss,
            )
        elseif !isnothing(best_adj)
            @info "Adaptation: existing-parameter fit ($(best_adj.name)) did not fully " *
                  "explain the drift (drift_detected=$(best_adj.drift_result.drift_detected), " *
                  "at_bound=$(best_adj.at_bound)) — falling through to structural search."
        end
    end

    # 2. One-time characterisation. Without a trained UDE the summary is empty
    # rather than a `_characterise_nn` error.
    nn_summary = include_ude_sections && !isnothing(update_result) ?
        _characterise_nn(update_result; input_ranges, n_samples) : ""

    sensor_summary = if !isnothing(data_times) && !isnothing(data_states) &&
                        !isempty(problem.observable_states)
        _characterise_sensor_residuals(problem, data_times, data_states)
    else
        ""
    end

    # Positions the measurements still permit, from the same balance laws the
    # characterisation uses, so prompt and analysis agree.
    allowed_hooks = Symbol[]
    if restrict_hooks_by_snr && !isnothing(data_states)
        snrs = hook_snrs(problem, data_times, data_states)
        allowed_hooks = [k for (k, v) in snrs if v >= hook_snr_min]
        isempty(allowed_hooks) &&
            (@warn "restrict_hooks_by_snr: no hook clears SNR $(hook_snr_min); " *
                   "falling back to the full list rather than offering none.")
    end

    # The sensor residuals say which channels deviate; this describes the
    # missing element at each hook. Without a UDE its sections are dropped.
    include_ude_sections = include_ude_sections && !isnothing(update_result)

    hook_summary = isnothing(hook_summary) ?
        _characterise_hooks(problem, data_times, data_states;
                            include_analysis = hook_local_analysis,
                            terminal_inputs = hook_terminal_inputs,
                            only_hooks = allowed_hooks) : String(hook_summary)

    # The per-hook relevance ranking was removed: it divided by a signal range
    # that is noise on a near-constant channel. See `_score_hooks_by_residuals`.


    # 3. Retry loop: prompt, call, fit, drift check, accept or retry. The
    # attempt with the lowest complexity-penalised loss is returned.

    default_ct = isempty(problem.component_guesses) ? nothing :
                                                       first(problem.component_guesses).component_type

    failed_attempt_texts = String[]
    tried_fingerprints   = Set{Any}()
    best_proposals       = ComponentProposal[]
    best_score           = Inf64   # complexity-penalised fit loss used for comparison
    best_fit_loss        = Inf64
    best_per_sensor      = Pair{String, Float64}[]
    best_raw_response    = "[no LLM call made]"
    best_prompt          = ""
    best_fit_ok          = false
    best_attempt_idx     = 0
    n_attempts_done      = 0
    # Set only on the `explain_on_failure` path; see the keyword's comment.
    no_library_fit       = false
    # Every prompt and response pair, including discarded attempts; the
    # top-level fields keep the accepted one only.
    conversation = LLMExchange[]

    max_attempts = max_retries + 1
    for attempt_idx in 1:max_attempts
        n_attempts_done += 1

        # 2a. Build prompt, with failure context added on retries.
        prompt = _build_adaptation_prompt(problem, update_result, nn_summary;
                                          sensor_summary, hook_summary,
                                          include_ude_sections, allowed_hooks,
                                          include_component_equations,
                                          failed_attempts = failed_attempt_texts)

        # 2b. Call LLM (or demo stub).
        raw_response, response_proposals, response_code = _call_llm(
            api, prompt, api_key;
            base_url, model, num_ctx, think,
            hooks    = problem.component_hooks,
            aliases  = problem.port_aliases,
            default_component_type = default_ct,
            log_file = llm_log_file,
            timeout  = llm_timeout,
        )
        push!(conversation, LLMExchange(attempt_idx, String(prompt), String(raw_response),
                                        length(response_proposals), "called"))

        # 2c. Validate LLM proposals.
        # Step i: reject proposals referencing subsystems not in the base model.
        subsys_valid = let
            valid_sys = Set(string(nameof(s))
                            for s in ModelingToolkit.get_systems(problem.model))
            filter(response_proposals) do prop
                a_ok = string(prop.port_a[1]) in valid_sys
                b_ok = string(prop.port_b[1]) in valid_sys
                if !a_ok || !b_ok
                    bad = filter(!in(valid_sys),
                                 [string(prop.port_a[1]), string(prop.port_b[1])])
                    @warn "Discarding LLM proposal '$(prop.name)': references unknown " *
                          "subsystem(s) $(join(repr.(bad), ", ")).  " *
                          "Valid: $(join(sort(collect(valid_sys)), ", "))"
                end
                a_ok && b_ok
            end
        end

        # Step ii: reject proposals whose port pair doesn't match any ComponentHook.
        valid_proposals = if isempty(problem.component_hooks)
            subsys_valid
        else
            hook_pairs = Set(minmax(h.port_a, h.port_b) for h in problem.component_hooks)
            filter(subsys_valid) do prop
                in_hooks = minmax(prop.port_a, prop.port_b) ∈ hook_pairs
                if !in_hooks
                    @warn "Discarding LLM proposal '$(prop.name)': port pair " *
                          "($(prop.port_a[1]).$(prop.port_a[2]) ↔ " *
                          "$(prop.port_b[1]).$(prop.port_b[2])) does not match any ComponentHook. " *
                          "Valid: $(join(["$(h.port_a[1]).$(h.port_a[2])↔$(h.port_b[1])." *
                                         "$(h.port_b[2])" for h in problem.component_hooks], ", "))"
                end
                in_hooks
            end
        end

        if !isempty(response_proposals) && isempty(valid_proposals)
            @warn "All $(length(response_proposals)) LLM proposal(s) were invalid — " *
                  "falling back to demo-mode enumeration."
        end

        # No usable proposal. With `explain_on_failure` the library is not
        # enumerated, which would turn a model failure into a wrong answer.
        if isempty(valid_proposals) && explain_on_failure && api !== :none
            no_library_fit = true
            # Keep the response: it is otherwise only stored after a
            # successful fit, and the failure evidence would be lost.
            best_raw_response = raw_response
            best_prompt       = prompt
            best_attempt_idx  = attempt_idx
            @info "Adaptation: the LLM produced no usable proposal and " *
                  "explain_on_failure is set — declining to fall back to demo-mode " *
                  "enumeration; an explanation will be produced instead."
            break
        end

        # 2d. Select proposals, deduplicated across attempts. Valid LLM
        # proposals win; in demo mode the attempt index drives enumeration.
        proposals_candidate = if !isempty(valid_proposals)
            valid_proposals
        else
            _proposals_from_guesses(problem.component_hooks, problem.component_guesses;
                                    attempt = attempt_idx)
        end

        fp = _proposal_fingerprint(proposals_candidate)
        if fp in tried_fingerprints
            # Already tried: advance the counter to a new combination. The
            # base is (n_guesses + 1), excluding the all-empty case.
            n_combos = (length(problem.component_guesses) + 1)^max(1, length(problem.component_hooks)) - 1
            found = false
            for alt in attempt_idx+1 : attempt_idx + n_combos
                alt_props = _proposals_from_guesses(problem.component_hooks,
                                                    problem.component_guesses; attempt = alt)
                isempty(alt_props) && continue   # all-skip combination
                alt_fp = _proposal_fingerprint(alt_props)
                if !(alt_fp in tried_fingerprints)
                    proposals_candidate = alt_props
                    fp = alt_fp
                    found = true
                    break
                end
            end
            if !found
                @info "Adaptation: all $(n_combos) component-type combination(s) exhausted — " *
                      "stopping retry loop after $n_attempts_done attempt(s)."
                break
            end
        end
        push!(tried_fingerprints, fp)
        proposals = proposals_candidate
        isempty(proposals) && break

        # 2e. Fit parameters + drift detection.
        fit_msg  = ""
        fit_loss = NaN
        fit_ok   = false
        drift_result = DriftResult(false, nothing, NaN, NaN, :none, Dict{Symbol,Any}())
        per_sensor   = Pair{String, Float64}[]
        bound_flags  = String[]
        if fit_params && !isnothing(data_times) && !isnothing(data_states) &&
                !isempty(proposals)
            # _winnow_proposals subsumes _fit_proposal_parameters and keeps the
            # smallest adequate subset, so `proposals` may come back shorter.
            proposals, fit_ok, fit_loss, drift_result, per_sensor, bound_flags =
                _winnow_proposals(
                    proposals, problem, data_times, data_states;
                    max_iters = fit_iters, lr = fit_lr,
                    holdout_fraction = Float64(holdout_fraction),
                    optimization_algorithm = optimization_algorithm
                )
            fit_msg = fit_ok ? " (parameters fitted to data)" :
                               " (parameter fitting failed — using defaults)"
        end

        # 2f. Track the best attempt by complexity-penalised score; one with a
        # non-finite loss is kept only while no finite-scored attempt exists.
        score = isfinite(fit_loss) ?
                fit_loss * (1 + Float64(complexity_penalty) * length(proposals)) : Inf64
        if score < best_score || (isempty(best_proposals) && !isempty(proposals))
            best_score        = score
            best_fit_loss     = fit_loss
            best_proposals    = proposals
            best_raw_response = raw_response
            best_prompt       = prompt
            best_fit_ok       = fit_ok
            best_attempt_idx  = attempt_idx
            # `per_sensor` is loop-local, so the accepted attempt is kept for
            # the structure-quality verdict after the loop.
            best_per_sensor   = per_sensor
        end

        # Record the outcome, including how many proposals survived winnowing,
        # which the raw response and the final result do not show.
        if !isempty(conversation) && last(conversation).attempt == attempt_idx
            ex = pop!(conversation)
            push!(conversation, LLMExchange(ex.attempt, ex.prompt, ex.raw_response, ex.n_parsed,
                "parsed $(ex.n_parsed) proposal(s); $(length(proposals)) kept after " *
                "validation/winnowing; fit_loss=$(round(fit_loss; sigdigits=4))"))
        end

        # 2g. Retry on remaining drift, on a channel above the declared sensor
        # noise, or on a parameter at a plausibility bound.
        ratio = NaN
        over  = String[]
        for (nm, rmse) in per_sensor
            j = findfirst(s -> string(s) == nm, string.(problem.observable_states))
            isnothing(j) && continue
            sigma = problem.observation_noise[j]
            sigma > 0 && rmse > _hl_accept_factor(length(data_times), 1) * sigma &&
                push!(over, "$nm ($(round(rmse/sigma; sigdigits=3))x sigma)")
        end
        structural_doubt = drift_result.drift_detected || !isempty(over)
        isempty(over) || @info "Structure fit quality: channel(s) above the sensor " *
                               "noise floor: " * join(over, ", ")
        should_retry = attempt_idx < max_attempts && isfinite(fit_loss) &&
                       (structural_doubt || !isempty(bound_flags))

        if !should_retry
            break
        end

        # Record failure and log so next iteration can augment the prompt.
        push!(failed_attempt_texts,
              _format_failed_attempt(attempt_idx, proposals, fit_loss, NaN,
                                     ratio, Float64(retry_ratio_threshold),
                                     drift_result, per_sensor, bound_flags))
        reason = structural_doubt ?
                 "adapted model still drifts" *
                 (isempty(over) ? "" : "; above sensor noise on " * join(over, ", ")) :
                 "fitted parameter(s) pinned at plausibility bounds"
        @info "Adaptation attempt $attempt_idx: $reason. Retrying with augmented " *
              "prompt ($(max_attempts - attempt_idx) attempt(s) remaining)."
    end

    # 4. Assemble final result from the best attempt
    proposals     = best_proposals
    fit_loss      = best_fit_loss
    raw_response  = best_raw_response
    prompt        = isempty(best_prompt) ?
                    _build_adaptation_prompt(problem, update_result, nn_summary;
                                             sensor_summary, hook_summary,
                                             include_ude_sections, allowed_hooks,
                                             include_component_equations) : best_prompt

    # Mark which exchange actually produced the returned result.
    if best_attempt_idx > 0
        for (i, ex) in enumerate(conversation)
            ex.attempt == best_attempt_idx || continue
            conversation[i] = LLMExchange(ex.attempt, ex.prompt, ex.raw_response,
                                          ex.n_parsed, "ACCEPTED — " * ex.note)
        end
    end

    # Structure-quality verdict against the declared sensor noise, which no
    # model can fit better than, unlike the former fit-to-UDE loss ratio.
    fit_verdict = ""
    worst, worst_ch = 0.0, ""   # visible after the block below: `escalate_on_inadequate_structure` reads it
    if isfinite(fit_loss)
        for (nm, rmse) in best_per_sensor
            k = findfirst(v -> string(v) == nm, string.(problem.observable_states))
            isnothing(k) && continue
            sigma = problem.observation_noise[k]
            sigma > 0 && rmse / sigma > worst && (worst = rmse / sigma; worst_ch = nm)
        end
        if worst > 0
            fit_verdict = if worst <= 1.5
                " — structure quality: good (every channel within $(round(worst; sigdigits=3))x its sensor noise)"
            elseif worst <= 5
                " — structure quality: partial (worst channel $(worst_ch) at " *
                "$(round(worst; sigdigits=3))x its sensor noise)"
            else
                " — structure quality: POOR (worst channel $(worst_ch) at " *
                "$(round(worst; sigdigits=3))x its sensor noise; structure likely incomplete)"
            end
            @info "Structure fit quality: worst channel $(worst_ch) at " *
                  "$(round(worst; sigdigits=3))x sensor noise" *
                  (worst > 5 ? "  ← STRUCTURE MAY BE INCOMPLETE" : "")
        end
    end

    # Generated from the final proposals, not from the returned
    # `julia_model_code`, which still wires discarded components.
    generated_code = _generate_adaptation_code(problem, proposals)

    # Optionally build the model.
    adapted_model = nothing
    build_msg     = ""
    if build_model
        adapted_model, build_msg = _try_build_model(generated_code)
    end

    # Terminal case of the retry loop, rejected rather than returned as a
    # success. See "Escalation on an inadequate structure" in the docs.
    fit_totally_failed = !best_fit_ok && !isfinite(fit_loss)
    structure_rejected = escalate_on_inadequate_structure && !isempty(proposals) &&
                         (worst > 5 || fit_totally_failed)

    success      = !isempty(proposals) && !structure_rejected
    attempt_note = n_attempts_done > 1 ? " ($n_attempts_done attempt(s))" : ""
    fit_note     = best_fit_ok ? " (parameters fitted to data)" : ""
    chosen_note  = join(["$(p.name)::$(p.component_type)" for p in proposals], ", ")
    message = if structure_rejected && fit_totally_failed
        "Model adaptation REJECTED: best structure ($chosen_note) could not be fitted " *
        "to the data at all after $(n_attempts_done) attempt(s) — escalating to a human."
    elseif structure_rejected
        "Model adaptation REJECTED: best structure ($chosen_note) still leaves " *
        "$(worst_ch) at $(round(worst; sigdigits=3))x sensor noise after " *
        "$(n_attempts_done) attempt(s)$(fit_note) — escalating to a human."
    elseif success
        "Model adaptation produced $(length(proposals)) proposal(s) " *
        "via api=$(api)$(fit_note)$(attempt_note)$(fit_verdict)" *
        (isempty(build_msg) ? "" : " — $build_msg")
    else
        "Model adaptation produced no proposals."
    end

    # Report the measured evidence in terms a human can act on. A separate
    # LLM call; the adaptation prompt above is unchanged.
    escalation = if structure_rejected
        offered = join([string(g.component_type) for g in problem.component_guesses], ", ")
        # The rationale of the rejected proposal, so the explanation can say
        # which evidence pointed at it and why that is insufficient.
        tried = join(["\"$(p.name)\" ($(p.component_type)) at " *
                     "$(p.port_a[1]).$(p.port_a[2]) ↔ $(p.port_b[1]).$(p.port_b[2]) — " *
                     "proposed because: $(p.rationale)" for p in proposals], "\n  ")
        live = isempty(sensor_summary) ? "" :
               "\n\nMeasured sensor residuals:\n" * sensor_summary
        hooks = isempty(hook_summary) ? "" :
                "\n\nPer-hook balance-law measurement:\n" * hook_summary
        # A build or fit failure has no worst-channel number to quote, so the
        # fit-quality sentence differs from the poor-fit case.
        fit_state = fit_totally_failed ?
            "the parameters for this fix could not be fitted to the data at all " *
            "(the attempt failed before producing any usable result)" :
            "the adapted twin still leaves $(worst_ch) at $(round(worst; sigdigits=3))x " *
            "its sensor noise floor"
        _expl = "A digital twin of \"$(problem.name)\" no longer matches its measurements. " *
                "After $(n_attempts_done) attempt(s), the best structural fix tried was:\n" *
                "  $tried\n" *
                "but $fit_state, and the retry budget is now exhausted. The only " *
                "component types available to try were: $offered.\n\n" *
                "Explain this to a non-expert. Return ONLY a JSON object (no markdown prose " *
                "outside the JSON) with exactly these two string fields:\n" *
                "  explanation    : what behaviour was found in the measurements, what it " *
                "likely indicates physically" *
                (fit_totally_failed ? "" :
                 " (name the specific channel that remains unexplained and by how far it misses)") *
                ", and why it does not match any of the available component types " *
                "($offered) — reason about the OTHER types too, not only the one that was " *
                "tried.\n" *
                "  recommendation : concrete, specific things the person reviewing this " *
                "should check or investigate next. Do NOT propose a specific new component " *
                "as if it were the answer — guide a human to investigate, don't hand them a fix.\n" *
                live * hooks
        # Explained by the backend that proposed the rejected structure, not
        # by the separate agent in SupervisionConfig.
        raw_expl = _call_llm_text(api, _expl, api_key; base_url, model, num_ctx, think,
                                  timeout = llm_timeout, log_file = llm_log_file)
        expl, rec, parsed_ok = _parse_escalation_json(raw_expl)
        # A malformed response is kept as the explanation field, with the
        # parse failure recorded in the recommendation field.
        parsed_ok ||
            @warn "Escalation explanation was not valid {explanation, recommendation} JSON " *
                  "— storing the raw response instead."
        txt = parsed_ok ? _escalation_to_json(expl, rec) :
              _escalation_to_json(raw_expl,
                  "The explanation above could not be parsed as JSON; review it directly.")
        @info "Adaptation: best structural fit " *
              (fit_totally_failed ? "could not be fitted at all" : "still POOR") *
              " after $(n_attempts_done) attempt(s)" *
              (fit_totally_failed ? "" : " (worst channel $(round(worst; sigdigits=3))x noise)") *
              " — escalating."
        txt
    elseif no_library_fit
        offered = join([string(g.component_type) for g in problem.component_guesses], ", ")
        live = isempty(sensor_summary) ? "" :
               "\n\nMeasured sensor residuals:\n" * sensor_summary
        hooks = isempty(hook_summary) ? "" :
                "\n\nPer-hook balance-law measurement:\n" * hook_summary
        _expl = "A digital twin of \"$(problem.name)\" no longer matches its " *
                "measurements. The diagnosis step could not explain the mismatch with " *
                "any of the component types it is allowed to propose ($offered).\n\n" *
                "Explain this to a non-expert. Return ONLY a JSON object (no markdown prose " *
                "outside the JSON) with exactly these two string fields:\n" *
                "  explanation    : what behaviour was found in the measurements below, where " *
                "the discrepancy is located, what it likely indicates physically, and why " *
                "none of the available component types ($offered) accounts for it.\n" *
                "  recommendation : concrete, specific things the person reviewing this " *
                "should check or investigate next. Do NOT invent a component as if it were " *
                "the answer — guide a human to investigate, don't hand them a fix.\n" *
                live * hooks
        # Same backend as the unusable answer (see `_call_llm_text`), with the
        # pipeline timeout: this prompt carries the full evidence.
        raw_expl = _call_llm_text(api, _expl, api_key; base_url, model, num_ctx, think,
                                  timeout = llm_timeout, log_file = llm_log_file)
        expl, rec, parsed_ok = _parse_escalation_json(raw_expl)
        parsed_ok ||
            @warn "Escalation explanation was not valid {explanation, recommendation} JSON " *
                  "— storing the raw response instead."
        txt = parsed_ok ? _escalation_to_json(expl, rec) :
              _escalation_to_json(raw_expl,
                  "The explanation above could not be parsed as JSON; review it directly.")
        @info "Adaptation: no offered component explains the measurement — escalating."
        txt
    else
        ""
    end

    return ModelAdaptationResult(
        proposals,
        ParameterAdjustment[],
        generated_code,
        adapted_model,
        prompt,
        raw_response,
        api,
        success,
        message,
        fit_loss,
        conversation,
        escalation,
    )
end

# --- Human-readable summary -------------------------------------------------

"""
    model_adaptation_summary(result::ModelAdaptationResult) -> String

Summarise a result: the proposed components, the parameter adjustments and the
generated code.

# Arguments
- `result`: the result to summarise.

# Returns
The summary as a multi-line string.
"""
function model_adaptation_summary(result::ModelAdaptationResult)::String
    io = IOBuffer()
    println(io, "=== AIRMED Model Adaptation ===")
    println(io, "API:        $(result.api)")
    println(io, "Status:     $(result.success ? "success" : "failed")")
    println(io, "Message:    $(result.message)")
    if isfinite(result.fit_loss)
        println(io, "Fit MSE:    $(round(result.fit_loss; sigdigits=4))  " *
                    "(RMSE = $(round(sqrt(result.fit_loss); sigdigits=4)))")
        # Also show ratio vs. UDE loss if the ModelAdaptationResult was created with
        # an UpdateResult available (stored indirectly via result.message or externally).
    end
    if !isempty(result.parameter_adjustments)
        println(io, "")
        println(io, "Parameter adjustments ($(length(result.parameter_adjustments))) — " *
                    "no structural change:")
        for (i, adj) in enumerate(result.parameter_adjustments)
            println(io, "  [$i] $(adj.name) : $(round(adj.old_value; sigdigits=4)) → " *
                        "$(round(adj.new_value; sigdigits=4))")
            isempty(adj.rationale) || println(io, "       rationale = $(adj.rationale)")
        end
    end
    println(io, "")
    println(io, "Proposals ($(length(result.proposals))):")
    for (i, p) in enumerate(result.proposals)
        pstr = join(["$k=$(round(v; sigdigits=4))" for (k, v) in p.parameters], ", ")
        a = "$(p.port_a[1]).$(p.port_a[2])"
        b = "$(p.port_b[1]).$(p.port_b[2])"
        topo = p.port_a[1] == p.port_b[1] ? "parallel with $(p.port_a[1])" : "series between $a and $b"
        println(io, "  [$i] $(p.name) :: $(p.component_type)")
        println(io, "       ports     = $a ↔ $b  ($topo)")
        println(io, "       params    = $pstr")
        isempty(p.rationale) || println(io, "       rationale = $(p.rationale)")
    end
    println(io, "")
    println(io, "Generated Julia/MTK code:")
    println(io, "----------------------------------------")
    print(io, result.generated_code)
    println(io)
    println(io, "----------------------------------------")
    result.adapted_model === nothing || println(io, "\nAdapted ODESystem was built successfully.")
    return String(take!(io))
end

# --- Internals: NN characterisation -----------------------------------------

function _characterise_nn(
    update_result::UpdateResult;
    input_ranges::Union{Nothing, Vector{<:Tuple{<:Real, <:Real}}} = nothing,
    n_samples::Int = 50,
)
    update_result.success || return "NN training did not succeed; no characterisation available."
    isnothing(input_ranges) && return "No input_ranges supplied; NN characterisation skipped."

    try
        inputs, outputs = symbolic_regression_of_nn(update_result, input_ranges; n_samples)
        n_in  = size(inputs, 1)
        n_out = size(outputs, 1)
        io    = IOBuffer()

        # Input ranges and output statistics
        println(io, "NN correction sampled over $(n_in) input(s):")
        for (d, r) in enumerate(input_ranges)
            println(io, "  input[$d] ∈ [$(r[1]), $(r[2])]")
        end
        println(io, "Output statistics over full sample grid:")
        for d in 1:n_out
            v = outputs[d, :]
            println(io, "  output[$d]: min=$(round(minimum(v); sigdigits=4))  " *
                        "max=$(round(maximum(v); sigdigits=4))  " *
                        "mean=$(round(mean(v); sigdigits=4))  " *
                        "std=$(round(std(v); sigdigits=4))")
        end

        # Linear symbolic regression
        println(io)
        println(io, "Linear regression (output ≈ c₀ + c₁·x₁ + c₂·x₂ + ...):")
        _print_linear_fit!(io, inputs, outputs)

        # STLSQ recovers nonlinear structure that the linear fit cannot, and
        # says more than minimum, maximum and mean statistics.
        try
            println(io)
            println(io, "Sparse symbolic regression (STLSQ, polynomial basis, R² close to 1 ⇒ trustworthy):")
            for (d, fit) in enumerate(sparse_regression(inputs, outputs))
                println(io, "  output[$d] ≈ $(fit.expression)   R²=$(round(fit.r2; sigdigits=3))")
            end
        catch err_sr
            println(io, "  (sparse regression failed: $(sprint(showerror, err_sr)))")
        end

        # Partial sensitivities at midpoint via central finite differences
        try
            println(io)
            println(io, "Partial sensitivities ∂output/∂input at input midpoint:")
            mids = Float64[(r[1] + r[2]) / 2 for r in input_ranges]
            for i in 1:n_in
                h  = Float64(max((input_ranges[i][2] - input_ranges[i][1]) * 0.005, 1e-5))
                xp = copy(mids); xp[i] += h
                xm = copy(mids); xm[i] -= h
                yp = _eval_nn_at(update_result, xp)
                ym = _eval_nn_at(update_result, xm)
                for d in 1:n_out
                    sens = Float64((yp[d] - ym[d]) / (2h))
                    println(io, "  ∂output[$d]/∂input[$i] ≈ $(round(sens; sigdigits=4))")
                end
            end
        catch err2
            println(io, "  (sensitivity analysis failed: $(sprint(showerror, err2)))")
        end

        # Monotonicity: sweep each input while holding others at midpoint
        try
            println(io)
            println(io, "Monotonicity (sweeping each input; others held at midpoint):")
            mids = Float64[(r[1] + r[2]) / 2 for r in input_ranges]
            for i in 1:n_in
                sweep = range(Float64(input_ranges[i][1]), Float64(input_ranges[i][2]); length=20)
                y_sweep = map(sweep) do v
                    x = copy(mids); x[i] = v
                    Float64.(_eval_nn_at(update_result, x))
                end
                for d in 1:n_out
                    vals  = [y[d] for y in y_sweep]
                    diffs = diff(vals)
                    mono  = if all(>=(-1e-8), diffs)
                        "monotone increasing"
                    elseif all(<=(1e-8), diffs)
                        "monotone decreasing"
                    else
                        "non-monotone"
                    end
                    println(io, "  output[$d] vs input[$i]: $mono")
                end
            end
        catch err3
            println(io, "  (monotonicity analysis failed: $(sprint(showerror, err3)))")
        end

        # Representative sample table: sweep input[1], others at midpoint
        try
            println(io)
            println(io, "Representative sample (input[1] swept; others at midpoint):")
            mids   = Float64[(r[1] + r[2]) / 2 for r in input_ranges]
            sweep1 = range(Float64(input_ranges[1][1]), Float64(input_ranges[1][2]); length=5)
            for v in sweep1
                x = copy(mids); x[1] = v
                y = _eval_nn_at(update_result, x)
                x_s = join(round.(Float64.(x); sigdigits=4), ", ")
                y_s = join(round.(Float64.(y); sigdigits=4), ", ")
                println(io, "  x=[$x_s] → y=[$y_s]")
            end
        catch err4
            println(io, "  (sample table failed: $(sprint(showerror, err4)))")
        end

        # Temporal signature at a 5% ratio: a correction still non-zero at the
        # end of the state range indicates dissipation, a decaying one storage.
        try
            println(io)
            println(io, "Temporal signature analysis (PERSISTENT vs TRANSIENT correction):")
            mids    = Float64[(r[1] + r[2]) / 2 for r in input_ranges]
            x_start = copy(mids); x_start[1] = Float64(input_ranges[1][1])
            x_end   = copy(mids); x_end[1]   = Float64(input_ranges[1][2])
            y_start = Float64.(_eval_nn_at(update_result, x_start))
            y_end   = Float64.(_eval_nn_at(update_result, x_end))
            for d in 1:length(y_start)
                ratio = abs(y_end[d]) / (abs(y_start[d]) + 1e-12)
                sig   = ratio > 0.05 ? "PERSISTENT" : "TRANSIENT"
                println(io, "  output[$d]: at input[1]=$(round(x_start[1];sigdigits=4)) → $(round(y_start[d];sigdigits=4))")
                println(io, "             at input[1]=$(round(x_end[1];sigdigits=4)) → $(round(y_end[d];sigdigits=4))")
                println(io, "             |end/start| = $(round(ratio; sigdigits=3)) → $sig")
                if sig == "PERSISTENT"
                    println(io, "    INTERPRETATION: the correction is still significantly non-zero at the end")
                    println(io, "    of the state range. This is the signature of an ENERGY-DISSIPATING component")
                    println(io, "    (one that introduces steady losses, not merely transient dynamics).")
                    println(io, "    Prefer component types described as energy-dissipating in the component list.")
                else
                    println(io, "    INTERPRETATION: the correction decays to near-zero by the end of the state")
                    println(io, "    range. This is the signature of an ENERGY-STORING component (one that affects")
                    println(io, "    only the transient phase, not the steady state).")
                    println(io, "    Prefer component types described as energy-storing in the component list.")
                end
            end
        catch err5
            println(io, "  (temporal signature analysis failed: $(sprint(showerror, err5)))")
        end

        return String(take!(io))
    catch err
        return "NN characterisation failed: $(sprint(showerror, err))"
    end
end

"""
    _characterise_sensor_residuals(problem, data_times, data_states) -> String

Simulate the base model and describe the residual of each observable channel,
as measurement minus prediction, for the prompt.

Besides the mean and maximum magnitude, the text reports the temporal trend per
channel, comparing the first and last fifth of the samples, and the initial rate
of the first dynamic state against the base model. A persistent residual
indicates dissipation, a decaying one storage, and the initial rate separates
added dissipation from added storage.

# Arguments
- `problem`: supplies the base model and the observable states.
- `data_times`, `data_states`: the measurements.

# Returns
A multi-line string, or an error message if the simulation fails.
"""
function _characterise_sensor_residuals(
    problem::AIRMEDProblem,
    data_times::AbstractVector,
    data_states::AbstractMatrix;
    solver = Rodas5P(),
)::String
    isempty(problem.observable_states) && return "No observable states defined."
    try
        # make_ode_problem forwards problem.adaptation_guesses, required for
        # models whose initialisation DAE has cyclic symbolic substitutions.
        base_prob = make_ode_problem(problem)
        base_sol  = solve(base_prob, solver;
                          saveat  = collect(Float64.(data_times)),
                          abstol  = 1e-8, reltol = 1e-8,
                          verbose = false)
        base_sol.retcode ∈ (ReturnCode.Success, ReturnCode.Default) ||
            return "Base model simulation failed (retcode=$(base_sol.retcode))."

        io = IOBuffer()
        println(io, "Sensor residuals  (measured − base_model_prediction):")
        println(io, "Positive → true system is ABOVE base model; Negative → BELOW.")
        println(io)

        # The initial rate of the first state encodes the dynamic opposition,
        # so against the base model it says whether the system is slower.
        try
            s0       = first(problem.observable_states)
            base_u0  = Float64.(base_sol[s0])
            meas_u0  = Float64.(data_states[1, :])
            n_pts    = length(data_times)
            if n_pts >= 4
                # Use central difference over the first few points for a stable estimate.
                n_fd      = min(4, n_pts - 1)
                dt        = Float64(data_times[n_fd+1]) - Float64(data_times[1])
                rate_data = (meas_u0[n_fd+1] - meas_u0[1]) / dt
                rate_base = (base_u0[n_fd+1] - base_u0[1]) / dt
                ratio_str = abs(rate_base) > 1e-12 ?
                            " (ratio data/base = $(round(rate_data / rate_base; sigdigits=4)))" : ""
                direction = rate_data > rate_base + 1e-12 ? "FASTER" :
                            rate_data < rate_base - 1e-12 ? "SLOWER" : "SAME"
                println(io, "Initial rate of $(s0):")
                println(io, "  data:      $(round(rate_data; sigdigits=4)) / time_unit")
                println(io, "  base model:$(round(rate_base; sigdigits=4)) / time_unit$ratio_str")
                println(io, "  → $direction initial response")
                if direction == "SLOWER"
                    println(io, "    A SLOWER initial rate means higher effective dynamic opposition at t=0.")
                    println(io, "    This is consistent with an energy-DISSIPATING element in series")
                    println(io, "    (increases total opposition) or in parallel with higher opposition")
                    println(io, "    than the original path (reduces combined throughput).")
                    println(io, "    It is NOT consistent with an energy-STORING element in parallel")
                    println(io, "    (which provides a low-opposition path at t=0, making the initial rate FASTER).")
                elseif direction == "FASTER"
                    println(io, "    A FASTER initial rate means lower effective dynamic opposition at t=0.")
                    println(io, "    This is consistent with an energy-STORING element added in parallel,")
                    println(io, "    or any low-opposition parallel path.")
                end
                println(io)
            end
        catch
        end

        # Per-channel signed residuals with temporal trend
        for (i, s) in enumerate(problem.observable_states)
            i > size(data_states, 1) && break
            try
                base_pred = Float64.(base_sol[s])
                measured  = Float64.(data_states[i, :])
                residuals = measured .- base_pred          # signed: meas − base

                n         = length(residuals)
                n_window  = max(1, div(n, 5))              # 20% window
                mean_res  = mean(residuals)
                max_abs   = maximum(abs, residuals)

                # Temporal trend: compare early (first 20%) vs late (last 20%).
                mean_early = mean(residuals[1:n_window])
                mean_late  = mean(residuals[max(1, n-n_window+1):n])
                trend_lbl, trend_note = if abs(mean_early) < 1e-12
                    "FLAT (near-zero throughout)", ""
                elseif abs(mean_late) < 0.1 * abs(mean_early)
                    "DECAYING",
                    "  Residual decays to near-zero → energy-STORING component signature. " *
                    "The effect disappears at steady state."
                elseif abs(mean_late) > 3.0 * abs(mean_early)
                    "GROWING",
                    "  Residual grows over time → the true system diverges increasingly " *
                    "from the base model (unusual; may indicate a feedback / nonlinearity)."
                else
                    "PERSISTENT",
                    "  Residual is sustained throughout → energy-DISSIPATING component " *
                    "signature. The effect is present at both transient and steady-state phases."
                end

                sign_str  = mean_res >  1e-12 ? "positive" :
                            mean_res < -1e-12 ? "negative" : "≈ zero"
                sig_range = maximum(measured) - minimum(measured)
                rel_pct   = sig_range > 1e-12 ?
                            round(100.0 * max_abs / sig_range; sigdigits=3) : NaN
                rel_str   = isnan(rel_pct) ? "" : "  ($(rel_pct)% of signal range)"

                println(io, "  $(s):")
                println(io, "    mean residual  = $(round(mean_res; sigdigits=4))  ($(sign_str))")
                println(io, "    max |residual| = $(round(max_abs; sigdigits=4))$(rel_str)")
                println(io, "    early mean     = $(round(mean_early; sigdigits=4))  " *
                            "late mean = $(round(mean_late; sigdigits=4))")
                println(io, "    temporal trend = $trend_lbl")
                isempty(trend_note) || println(io, trend_note)
            catch
                println(io, "  $(s): (could not evaluate)")
            end
        end
        return String(take!(io))
    catch err
        return "Sensor residual analysis failed: $(sprint(showerror, err))"
    end
end

"""
    _score_hooks_by_residuals(problem, data_times, data_states) -> Vector{Pair{Symbol,Float64}}

!!! warning "Not used in the prompt: unsound normalisation"
    The score divides by the signal range of each channel, which on a
    near-constant channel is observation noise, so a constant source can
    outrank the hook at the fault. Kept for reference; it needs a noise-aware
    denominator before it can return to the prompt.

Score each hook by the deviation between measurement and base-model prediction
at the subsystems it connects.

# Arguments
- `problem`: supplies the base model, hooks and observable states.
- `data_times`, `data_states`: the measurements.

# Returns
`(hook_name => score)` pairs, sorted descending, where the score is the maximum
relative residual over the connected subsystems as a percentage of the signal
range. The maximum rather than the sum, so a subsystem with many channels is not
favoured. Empty if the simulation fails or no observables exist.
"""
function _score_hooks_by_residuals(
    problem::AIRMEDProblem,
    data_times::AbstractVector,
    data_states::AbstractMatrix;
    solver = Rodas5P(),
)::Vector{Pair{Symbol,Float64}}
    (isempty(problem.observable_states) || isempty(problem.component_hooks)) &&
        return Pair{Symbol,Float64}[]
    try
        base_prob = make_ode_problem(problem)
        base_sol  = solve(base_prob, solver;
                          saveat  = collect(Float64.(data_times)),
                          abstol  = 1e-8, reltol = 1e-8,
                          verbose = false)
        base_sol.retcode ∈ (ReturnCode.Success, ReturnCode.Default) ||
            return Pair{Symbol,Float64}[]

        subsys_score = Dict{Symbol, Float64}()
        for (i, s) in enumerate(problem.observable_states)
            i > size(data_states, 1) && break
            s isa ModelingToolkit.AbstractSystem && continue
            try
                base_pred = Float64.(base_sol[s])
                measured  = Float64.(data_states[i, :])
                max_abs   = maximum(abs, measured .- base_pred)
                sig_range = maximum(measured) - minimum(measured)
                rel_pct   = sig_range > 1e-12 ? 100.0 * max_abs / sig_range : 0.0

                subsys = Symbol(first(split(string(ModelingToolkit.getname(s)), "₊")))
                subsys_score[subsys] = max(get(subsys_score, subsys, 0.0), rel_pct)
            catch
            end
        end
        isempty(subsys_score) && return Pair{Symbol,Float64}[]

        hook_scores = [h.name => maximum(get(subsys_score, s, 0.0) for s in (h.port_a[1], h.port_b[1]))
                       for h in problem.component_hooks]
        return sort(hook_scores; by = last, rev = true)
    catch
        return Pair{Symbol,Float64}[]
    end
end


# --- Hook-local characterisation ---------------------------------------------
# Measures the constitutive law of the missing element at each declared hook.

_hl_smooth(y, w) = (n = length(y); h = w ÷ 2;
                    [mean(@view y[max(1, i-h):min(n, i+h)]) for i in 1:n])

# Wide stencil: a one-step difference amplifies sensor noise beyond the true
# derivative; widening by k divides that by k and leaves smooth signal intact.
_hl_deriv(t, y; k = 10) = (n = length(y);
    [(y[min(n, i+k)] - y[max(1, i-k)]) / (t[min(n, i+k)] - t[max(1, i-k)]) for i in 1:n])

function _hl_cumtrapz(t, y)
    out = zeros(length(y))
    for i in 2:length(y)
        out[i] = out[i-1] + 0.5 * (y[i] + y[i-1]) * (t[i] - t[i-1])
    end
    return out
end

"""
    hook_snrs(problem, data_times, data_states; smooth_w) -> Dict{Symbol,Float64}

Balance-law signal-to-noise ratio at every declared hook: sum the response
terms with their signs, propagate the sensor variances, and divide by the
smoothed noise floor. Public, so a caller can restrict the prompt to the
positions the measurements support.

# Arguments
- `problem`: supplies `hook_residuals` and `observation_noise`.
- `data_times`, `data_states`: the measurements.

# Keywords
- `smooth_w`: width of the moving average. Default `15`.

# Returns
One SNR per hook name. A value near 1 means the law closes within noise.
"""
function hook_snrs(problem::AIRMEDProblem, data_times, data_states;
                   smooth_w::Int = 15)::Dict{Symbol,Float64}
    out = Dict{Symbol,Float64}()
    (isempty(problem.hook_residuals) || isempty(problem.observation_noise) ||
     isnothing(data_states)) && return out
    row = Dict(string(s) => i for (i, s) in enumerate(problem.observable_states))
    idx = v -> get(row, string(v), 0)
    for spec in problem.hook_residuals
        any(idx(k) == 0 for (k, _) in spec.response_terms) && continue
        resp = zeros(size(data_states, 2)); var_sum = 0.0
        for (k, c) in spec.response_terms
            j = idx(k)
            resp .+= c .* Float64.(data_states[j, :])
            var_sum += (c * problem.observation_noise[j])^2
        end
        y = _hl_smooth(resp, smooth_w)
        out[spec.hook] = sqrt(mean(abs2, y)) / max(sqrt(var_sum) / sqrt(smooth_w), 1e-30)
    end
    return out
end

"""
    _hl_operating_point!(io, d, y, floor_σ)

Report the single operating point that remains when nothing in the window
varies, with its standard error and the equivalence class of laws through it,
since one point cannot identify the form.

# Arguments
- `io`: buffer to write to.
- `d`, `y`: the drive and response series.
- `floor_σ`: the noise floor of the response.

# Returns
`nothing`; the report is written to `io`.
"""
function _hl_operating_point!(io, d, y, floor_σ)
    n      = length(y)
    ū, ȳ  = mean(d), mean(y)
    su, sy = std(d) / sqrt(n), std(y) / sqrt(n)
    println(io, "    operating point: drive = $(round(ū; sigdigits = 6)) ± " *
                "$(round(su; sigdigits = 2)),  response = $(round(ȳ; sigdigits = 6)) ± " *
                "$(round(sy; sigdigits = 2))   (n = $n)")
    println(io, "    drive · response at that point: $(round(ū * ȳ; sigdigits = 5))")
    println(io, "    FORM NOT IDENTIFIABLE from a single operating point — every law")
    println(io, "    through one point fits it equally well. Three of them, stated only")
    println(io, "    as relations between the measured drive and response:")
    println(io, "        response ∝ drive      →  drive / response = $(round(ū / ȳ; sigdigits = 5))")
    println(io, "        drive · response = c  →  c = $(round(ū * ȳ; sigdigits = 5))")
    println(io, "        response = c          →  c = $(round(ȳ; sigdigits = 5))")
    println(io, "    Which of these is physically right is a property of the COMPONENT, not")
    println(io, "    of this data — decide it from the component types offered above.")
end

"""
    _hl_basis_predict(fit, X) -> Vector

Evaluate a `sparse_regression` result on its own basis columns, so the post-fit
residual can be compared with the sensor noise floor.

The surviving terms are reported by name, so each one is looked up in the basis
rather than parsed. Parsing would fail on any non-monomial term, such as the
sin/cos pair the periodicity check adds.

# Arguments
- `fit`: one entry returned by `sparse_regression`.
- `names`, `Φ`: names and columns from the same `_build_basis` call the fit
  used, `extra_basis` included.

# Returns
The predicted response.
"""
function _hl_basis_predict(fit, names::AbstractVector{<:AbstractString},
                           Φ::AbstractMatrix)
    pred = zeros(size(Φ, 1))
    for (name, c) in fit.terms
        j = findfirst(==(String(name)), String.(names))
        isnothing(j) && error("basis term $(repr(name)) is not in the basis it was fitted on")
        pred .+= c .* @view Φ[:, j]
    end
    return pred
end

"""
    _hl_accept_factor(n, w; z = 2.326) -> Float64

Factor by which the residual of a correct law may exceed the noise floor.

Smoothing makes neighbouring samples dependent, leaving about n/w independent
samples rather than n, so RMSE/sigma has a sampling spread even when the law is
exact. The factor is the 99th percentile of sqrt(chi^2_k / k) under the
Wilson-Hilferty approximation; a strict test at 1.0 would reject the correct
component in about half of the cases.

# Arguments
- `n`: number of samples.
- `w`: smoothing width, so that `n/w` samples are independent.

# Keywords
- `z`: normal quantile behind the bound. Default `2.326`, the 99th percentile.

# Returns
The factor, slightly above 1 and approaching it as `n/w` grows.
"""
function _hl_accept_factor(n::Int, w::Int; z = 2.326)
    k = max(n / w, 1.0)
    return sqrt((1 - 2 / (9k) + z * sqrt(2 / (9k)))^3)
end

"""
    _hl_static_test(X, y, floor_σ; min_gap) -> (frac_explained, memory_rmse)

Test whether the residual is a static function of the measured channels, by
comparing the responses of nearest neighbours in input space.

If two instants look identical to every sensor while the element behaves
differently, no static law can describe it: the element has internal state or
depends on something unmeasured. No goodness-of-fit value reveals this.

# Arguments
- `X`: candidate inputs, one column per sample.
- `y`: the response.
- `floor_σ`: the noise floor of the response.

# Keywords
- `min_gap`: neighbours closer than this many samples in time are skipped, since
  smoothing correlates them.

# Returns
`(frac_explained, memory_rmse)`: the share of response variance the inputs
explain, and the RMS response difference between sensor-identical instants, in
physical units.
"""
function _hl_static_test(X::AbstractMatrix, y::AbstractVector, floor_σ;
                         min_gap::Int = 20)
    n = length(y)
    μ = [mean(@view X[j, :]) for j in axes(X, 1)]
    σ = [max(std(@view X[j, :]), 1e-12) for j in axes(X, 1)]
    Xn = (X .- μ) ./ σ
    diffs = Float64[]
    for i in 1:n
        best, bj = Inf, 0
        for j in 1:n
            abs(i - j) < min_gap && continue
            dist = 0.0
            for k in axes(Xn, 1); dist += (Xn[k, i] - Xn[k, j])^2; end
            dist < best && (best = dist; bj = j)
        end
        bj == 0 || push!(diffs, abs(y[i] - y[bj]))
    end
    isempty(diffs) && return (NaN, NaN)
    # Two sensor-identical instants differ by measurement noise on BOTH, so the
    # expected difference under a static law is √2·floor_σ, not floor_σ.
    mem = sqrt(mean(abs2, diffs))
    var_y = var(y)
    frac  = var_y <= 0 ? NaN : max(0.0, 1 - (mem^2 / 2) / var_y)
    return (frac, mem)
end

"""
    _hl_mode_analysis(t, y, floor_σ) -> NamedTuple | nothing

Split the response into two levels, maximising the between-class variance, and
report whether it switches rather than varying continuously. A duty-cycled load
is described better by its levels, duty ratio and period than by a fitted
coefficient.

# Arguments
- `t`, `y`: time and response series.
- `floor_σ`: the noise floor of the response.

# Returns
`(lo_lvl, hi_lvl, sep, duty, transitions, period)`, or `nothing` if the two
levels are not separated well above the noise.
"""
function _hl_mode_analysis(t, y::AbstractVector, floor_σ)
    n = length(y)
    n < 8 && return nothing
    s = sort(y)
    best_var, thr = -Inf, s[1]
    for k in 2:(n - 1)
        w1, w2 = k / n, 1 - k / n
        m1, m2 = mean(@view s[1:k]), mean(@view s[k+1:end])
        b = w1 * w2 * (m1 - m2)^2
        b > best_var && (best_var = b; thr = (s[k] + s[k+1]) / 2)
    end
    hi = y .> thr
    (all(hi) || !any(hi)) && return nothing
    lo_lvl, hi_lvl = mean(y[.!hi]), mean(y[hi])
    sep = (hi_lvl - lo_lvl) / max(floor_σ, 1e-30)
    # Below ~6x noise the "two levels" are just the tails of one noisy mode.
    sep < 6 && return nothing
    trans = count(i -> hi[i] != hi[i+1], 1:(n-1))
    return (; lo_lvl, hi_lvl, sep, duty = count(hi) / n, transitions = trans,
            period = trans >= 2 ? 2 * (last(t) - first(t)) / trans : NaN)
end

"""
    _hl_dominant_period(t, y; min_cycles = 2, peak_ratio = 8) -> Float64

Period of the strongest periodic component of `y`, from a periodogram of the
linearly detrended response. Detrending prevents a slow drift from dominating
the low bins.

The sparse fit uses degree-1 monomials, which approximate a periodic residual by
a ramp, so without this an oscillating element looks static. The period comes
from the residual itself, never from the true system.

# Arguments
- `t`, `y`: time and response series.

# Keywords
- `min_cycles`: lowest admissible number of cycles in the window. Default `2`.
- `peak_ratio`: how far the peak must stand above the median bin. Default `8`.

# Returns
The period in the units of `t`, or `NaN` if no line stands out.
"""
function _hl_dominant_period(t, y::AbstractVector; min_cycles::Int = 2,
                             peak_ratio::Real = 8.0)
    n = length(y)
    (n < 32 || length(t) != n) && return NaN
    tt = Float64.(collect(t))
    yy = Float64.(collect(y))
    A  = hcat(ones(n), tt)
    r  = yy .- A * (A \ yy)
    all(iszero, r) && return NaN
    # Below 4 samples per cycle a "line" is aliasing, not a periodicity.
    kmax = n ÷ 4
    kmax < min_cycles && return NaN
    pow = [abs2(sum(r[j] * cispi(-2 * k * (j - 1) / n) for j in 1:n))
           for k in min_cycles:kmax]
    pk = argmax(pow)
    # A peak that is not well clear of the typical bin is noise, not a period.
    pow[pk] < peak_ratio * max(median(pow), 1e-30) && return NaN
    k    = float(min_cycles + pk - 1)
    # The coarse DFT grid leaves a frequency error that accumulates phase.
    # Parabolic interpolation over three log-power bins recovers the position.
    if 1 < pk < length(pow)
        a, b, c = log.(max.((pow[pk-1], pow[pk], pow[pk+1]), 1e-300))
        denom = a - 2b + c
        if denom < 0                                  # a genuine maximum
            δ = 0.5 * (a - c) / denom
            isfinite(δ) && abs(δ) <= 0.5 && (k += δ)
        end
    end
    span = (tt[end] - tt[1]) * n / max(n - 1, 1)
    return span / k
end


# --- Internals: prompt construction -----------------------------------------


"""
    _characterise_hooks(problem, data_times, data_states; include_analysis=true) -> String

Per-hook constitutive measurement, formatted for the prompt. Every step is
classical numerics on the measured channels: a moving average, a
nearest-neighbour test for hidden state, an Otsu split, a periodogram and
sequentially-thresholded least squares. No network is involved.

# Arguments
- `problem`: supplies `hook_residuals`, `observation_noise` and the observables.
- `data_times`, `data_states`: the measurements.

# Keywords
- `include_analysis`: with `false`, only the balance-law residual and its SNR
  are reported. Default `true`.
- `smooth_w`: width of the moving average. Default `15`.
- `only_hooks`: restrict the report to these hooks.
- `terminal_inputs`: fit in the element's own terminal quantities.

# Returns
A multi-line string, empty when the problem declares no hook residuals or no
observation noise.
"""
function _characterise_hooks(problem::AIRMEDProblem, data_times, data_states;
                             include_analysis::Bool = true, smooth_w::Int = 15,
                             only_hooks::Vector{Symbol} = Symbol[],
                             # Restrict the static test and sparse fit to the
                             # element's own terminals; see `cand_i`.
                             terminal_inputs::Bool = false)
    (isempty(problem.hook_residuals) || isempty(problem.observation_noise) ||
     isnothing(data_times) || isnothing(data_states)) && return ""

    row = Dict(string(s) => i for (i, s) in enumerate(problem.observable_states))
    idx = v -> get(row, string(v), 0)

    io = IOBuffer()
    println(io, "  Each missing element's own (drive, response) pair was recovered DIRECTLY")
    println(io, "  FROM THE SENSORS via the balance law that must hold at its hook. No")
    println(io, "  simulation and no neural network is involved in that step — it is arithmetic")
    println(io, "  on measured channels. Residuals are quoted as multiples of the propagated")
    println(io, "  SENSOR NOISE FLOOR: at or below 1x means the law explains everything except")
    println(io, "  measurement noise, so nothing further is inferable from this data.")
    println(io)

    # With a restricted hook set only those are characterised; describing an
    # inactive hook would reintroduce the position just removed.
    specs = isempty(only_hooks) ? problem.hook_residuals :
            filter(sp -> sp.hook in only_hooks, problem.hook_residuals)
    for spec in specs
        di = idx(spec.drive)
        di == 0 && (println(io, "  $(spec.hook): drive channel not among observable_states — skipped\n");
                    continue)
        any(idx(k) == 0 for (k, _) in spec.response_terms) &&
            (println(io, "  $(spec.hook): a response channel is not among observable_states — skipped\n");
             continue)

        drive_raw = Float64.(data_states[di, :])
        resp_raw  = zeros(size(data_states, 2))
        var_sum   = 0.0
        for (k, c) in spec.response_terms
            j = idx(k)
            resp_raw .+= c .* Float64.(data_states[j, :])
            var_sum  += (c * problem.observation_noise[j])^2   # independent errors: variances add
        end

        d       = _hl_smooth(drive_raw, smooth_w)
        dd      = _hl_deriv(data_times, d)
        y       = _hl_smooth(resp_raw, smooth_w)

        # Candidate inputs: the observable channels, time, and the derivative
        # and integral of the drive. See "Choice of candidate inputs".
        used = Set{Int}()
        for (k, _) in spec.response_terms; push!(used, idx(k)); end
        di in used && @warn "Hook $(spec.hook): drive is also a response term; " *
                            "excluding it to keep the target non-trivial."
        cand_i    = terminal_inputs ? [j for j in (di,) if !(j in used)] :
                    [j for j in axes(data_states, 1) if !(j in used)]
        cand_name = [string(problem.observable_states[j]) for j in cand_i]
        cand_col  = Vector{Vector{Float64}}(
            [_hl_smooth(Float64.(data_states[j, :]), smooth_w) for j in cand_i])
        push!(cand_name, "t");  push!(cand_col, Float64.(collect(data_times)))
        # Gated on the variation of the drive itself: on a pinned rail both
        # are noise, whose own spread would pass any threshold.
        drive_live = std(d) > 3 * problem.observation_noise[di] / sqrt(smooth_w)
        if drive_live
            push!(cand_name, "d($(spec.drive))/dt"); push!(cand_col, dd)
            push!(cand_name, "∫$(spec.drive)dt")
            push!(cand_col, _hl_cumtrapz(data_times, d))
        end
        floor_σ = sqrt(var_sum) / sqrt(smooth_w)               # averaging w samples: σ/√w
        snr     = sqrt(mean(abs2, y)) / max(floor_σ, 1e-30)
        accept  = _hl_accept_factor(length(y), smooth_w)

        println(io, "  $(spec.hook)  [$(spec.kind)]" *
                    (isempty(spec.description) ? "" : " — $(spec.description)"))
        println(io, "    drive    = $(spec.drive)")
        println(io, "    response = " * join(["$(c > 0 ? "+" : "-")$(k)"
                                              for (k, c) in spec.response_terms], " "))
        println(io, "    signal / sensor-noise floor (SNR) = $(round(snr; sigdigits = 3))")

        if snr < 3
            println(io, "    → residual is AT the noise floor; nothing is missing at this hook.\n")
            continue
        end
        if !include_analysis
            println(io, "    → a residual of $(round(sqrt(mean(abs2, y)); sigdigits = 4)) is present, " *
                        "well above the noise floor.\n")
            continue
        end

        # Keep only inputs that vary, against the smoothed noise floor
        # sigma/sqrt(w); the raw sigma would discard channels with signal.
        live = [j for j in eachindex(cand_col)
                if std(cand_col[j]) > 3 * (j <= length(cand_i) ?
                                           problem.observation_noise[cand_i[j]] /
                                           sqrt(smooth_w) : 0.0)]
        dead = setdiff(eachindex(cand_col), live)
        isempty(dead) ||
            println(io, "    constant over this window (dropped): " *
                        join(cand_name[dead], ", "))

        if isempty(live)
            println(io, "    NO input channel varies — only the operating point is " *
                        "measurable here.\n")
            _hl_operating_point!(io, d, y, floor_σ)
            continue
        end
        println(io, "    varying inputs used: " * join(cand_name[live], ", "))

        X = reduce(vcat, [permutedims(cand_col[j]) for j in live])

        # Whether a static law exists at all, asked before fitting: no
        # goodness-of-fit number reveals an element with internal state.
        frac, mem = try
            _hl_static_test(X, y, floor_σ)
        catch err
            @warn "Hook $(spec.hook): static-function test failed: $(sprint(showerror, err))"
            (NaN, NaN)
        end
        if !isnan(mem)
            println(io, "    static-law check: sensor-identical instants differ by " *
                        "$(round(mem / floor_σ; sigdigits = 3))x noise")
            if mem > 3 * floor_σ
                # No domain examples: naming plausible hidden quantities would
                # hand the model the component class it must infer.
                println(io, "        → NOT a static function of the measured channels. The " *
                            "element has INTERNAL STATE, or depends on a")
                println(io, "        quantity that is not among the measured channels. " *
                            "No law in these variables — linear, polynomial or neural —")
                println(io, "        can represent it; the fit below describes only its average " *
                            "behaviour.")
            else
                println(io, "        → consistent with a static law in these channels " *
                            "(inputs account for $(round(100 * frac; sigdigits = 3))% of the " *
                            "response variance).")
            end
        end

        # Switching behaviour: two levels and a period describe a duty-cycled
        # load better than a fitted coefficient, and are directly usable.
        modes = try
            _hl_mode_analysis(data_times, y, floor_σ)
        catch err
            @warn "Hook $(spec.hook): mode analysis failed: $(sprint(showerror, err))"
            nothing
        end
        if !isnothing(modes)
            println(io, "    SWITCHING: the response sits at two distinct levels, " *
                        "$(round(modes.sep; sigdigits = 3))x noise apart:")
            println(io, "        low  = $(round(modes.lo_lvl; sigdigits = 5))   " *
                        "high = $(round(modes.hi_lvl; sigdigits = 5))   " *
                        "step = $(round(modes.hi_lvl - modes.lo_lvl; sigdigits = 5))")
            println(io, "        duty = $(round(100 * modes.duty; sigdigits = 3))% at the high " *
                        "level, $(modes.transitions) transition(s)" *
                        (isnan(modes.period) ? " (single switch — not periodic)" :
                         ", cycle period ≈ $(round(modes.period; sigdigits = 4))"))
            # States what the residual does and stops there, rather than
            # naming the component class the model must identify.
            println(io, "        A single transition is one switch event; repeated " *
                        "transitions mean the element alternates")
            println(io, "        between these two levels across the window.")
        end

        # Collinear inputs cannot be told apart, and on a slowly-draining pack
        # SoC is very nearly a ramp in t. Say so rather than let the fit pick one.
        for a in eachindex(live), b in eachindex(live)
            a < b || continue
            ca, cb = cand_col[live[a]], cand_col[live[b]]
            r = abs(cov(ca, cb) / max(std(ca) * std(cb), 1e-30))
            r > 0.99 && println(io, "    NOTE: $(cand_name[live[a]]) and $(cand_name[live[b]]) " *
                                    "are collinear here (|r| = $(round(r; digits = 4))) — a " *
                                    "dependence on one is indistinguishable from the other.")
        end

        # Generic-basis symbolic regression at degree 1, with a sin/cos pair on
        # a dominant spectral line. See the section of that name in the docs.
        extra  = Pair{String, Function}[]
        it_pos = findfirst(j -> cand_name[j] == "t", live)
        T_per  = isnothing(it_pos) ? NaN : _hl_dominant_period(data_times, y)
        Tr     = isfinite(T_per) ? round(T_per; sigdigits = 4) : NaN
        if !isnothing(it_pos) && isfinite(T_per)
            push!(extra, "sin(2π·t/$Tr)" => (x -> sin(2π * x[it_pos] / T_per)))
            push!(extra, "cos(2π·t/$Tr)" => (x -> cos(2π * x[it_pos] / T_per)))
        end

        fit = try
            # `scale_columns = true` is required: these channels differ in unit
            # and magnitude, so unscaled pruning discards terms by unit.
            first(sparse_regression(X, permutedims(y);
                                    basis_degree = 1, extra_basis = extra,
                                    scale_columns = true))
        catch err
            println(io, "    (sparse regression failed: $(sprint(showerror, err)))
")
            continue
        end

        # `sparse_regression` names inputs x1..xd positionally; restore the
        # channel names so the LLM reads measured quantities, not placeholders.
        pretty = fit.expression
        for (k, jj) in enumerate(live)
            pretty = replace(pretty, "x$(k)" => cand_name[jj])
        end

        # F-test against the intercept-only null, on the effective sample size
        # n/smooth_w. See "Significance of the fit" in the documentation.
        n_obs   = max(length(y) / smooth_w, 4.0)
        n_terms = max(count(!=(0.0), last.(fit.terms)) - 1, 0)   # excluding intercept
        f_stat  = (n_terms == 0 || n_obs - n_terms - 1 <= 0) ? 0.0 :
                  (fit.r2 / max(1 - fit.r2, 1e-12)) * (n_obs - n_terms - 1) / n_terms
        # F(n_terms, n-n_terms-1) at α=0.05 is ≈4 for these sample sizes; a plain
        # threshold avoids pulling in a distribution dependency for one test.
        significant = n_terms > 0 && f_stat > 4.0

        println(io, terminal_inputs ?
                    "    sparse fit of the response as a law in this element's OWN drive and time" :
                    "    sparse fit of the residual on the measured channels")
        println(io, "    (generic basis — no component assumed):")
        if significant
            println(io, "        response ≈ $(pretty)")
            println(io, "        R² = $(round(fit.r2; sigdigits = 4))   " *
                        "(F = $(round(f_stat; sigdigits = 3)) on $(n_terms) basis term(s))")
        else
            println(io, "        response ≈ $(round(mean(y); sigdigits = 6))   [CONSTANT]")
            println(io, "        No channel term is significant (R² = " *
                        "$(round(fit.r2; sigdigits = 3)), F = $(round(f_stat; sigdigits = 3)) " *
                        "against the constant model). The residual does not depend on any")
            println(io, "        measured channel or on time — consistent with a STATIC element.")
        end
        # Reported only if the periodic pair survived the fit; see the note
        # where `extra` is built.
        if significant && any(startswith(String(n), "sin(") || startswith(String(n), "cos(")
                              for (n, _) in fit.terms)
            println(io, "        The sin/cos period above was measured from THIS residual's own " *
                        "spectrum, and survived")
            println(io, "        pruning against every other candidate: the response OSCILLATES " *
                        "rather than being")
            println(io, "        constant or drifting.")
        end

        # The basis the fit was built on, `extra` included, so a surviving
        # sin/cos term is evaluated rather than parsed.
        bnames, Φb = _build_basis(Float64.(X), 1, extra)
        resid = y .- _hl_basis_predict(fit, bnames, Φb)
        rr    = sqrt(mean(abs2, resid)) / floor_σ
        println(io, "    residual after the fit: $(round(rr; sigdigits = 3))x the noise floor " *
                    "(<= $(round(accept; sigdigits = 3))x ⇒ fully explained)")
        # Condition number of the SCALED design matrix: the guard against reading
        # a fitted coefficient that the excitation cannot actually resolve.
        basis_cond = let
            σb = [max(sqrt(mean(abs2, @view Φb[:, j])), 1e-12) for j in axes(Φb, 2)]
            cond(Φb ./ σb')
        end
        println(io, "    identifiability: basis condition number " *
                    "$(round(basis_cond; sigdigits = 3))" *
                    (basis_cond > 1e4 ? "  — POORLY CONDITIONED, treat coefficients as indicative only" : ""))
        # The operating point is always reported: a stiff bus pins it down,
        # and it sizes a component even without a resolvable form.
        _hl_operating_point!(io, d, y, floor_σ)
        rr > accept && println(io, "    NOTE: these channels do not explain the residual — " *
                                   "the missing element may depend on something not measured, " *
                                   "or on time in a way this basis cannot express.")
        println(io)
    end
    return String(take!(io))
end

"""
    _component_law_lines(g::ComponentGuess) -> Vector{String}

The constitutive equations of one library component, as strings for the prompt.
The component is instantiated through its own `factory`, so the prompt shows the
equations of the component the framework would build.

Equations referencing a pin are boilerplate, identical for every two-pin
component, and are dropped; the prose states them once. State equations are
kept, since they are what distinguishes a storage element from a dissipative
one.

# Arguments
- `g`: the component to describe.

# Returns
One string per equation.
"""
function _component_law_lines(g::ComponentGuess)::Vector{String}
    try
        sys = g.factory(:this, g.default_parameter)
        lines = String[]
        for eq in equations(sys)
            s = replace(string(eq), "this₊" => "")
            # Pin-referencing equations are the OnePort trio; skip them.
            (occursin("p₊", s) || occursin("n₊", s)) && continue
            push!(lines, s)
        end
        return lines
    catch err
        @debug "Could not read equations for $(g.component_type)" exception = err
        return String[]
    end
end

"""
    _build_adaptation_prompt(problem, update_result, nn_summary;
                             sensor_summary="") -> String

Produce the natural-language prompt sent to the LLM. The prompt asks for a
JSON response with the following schema:

```json
{
    "proposals": [
        {
            "hook": "<ComponentHook name from problem.component_hooks>",
            "name": "<julia identifier for the new component>",
            "component_type": "<one of problem.component_guesses[].component_type>",
            "rationale": "..."
        }
    ],
    "julia_model_code": "...optional, MTK source for the adapted system..."
}
```

Parameter values are not requested; they are identified by optimisation.

# Keywords
- `sensor_summary`, `hook_summary`: pre-computed sections; `""` omits them.
- `allowed_hooks`: restrict the offered hook names.
- `include_ude_sections`: include the UDE architecture, loss and
  characterisation.
- `include_component_equations`: list the constitutive equations of each type.
- `failed_attempts`: texts of earlier attempts, added on a retry.

The prompt carries no per-hook relevance ranking; see
`_score_hooks_by_residuals` for why.

# Returns
The prompt as a string.
"""
function _build_adaptation_prompt(
    problem::AIRMEDProblem,
    # `Nothing` without a UDE. Every use sits inside the
    # `include_ude_sections` block, which is then forced false.
    update_result::Union{UpdateResult, Nothing},
    nn_summary::AbstractString;
    sensor_summary::AbstractString = "",
    hook_summary::AbstractString = "",
    failed_attempts::Vector{String} = String[],
    # With false, every section describing the global UDE is omitted, for
    # pipelines that diagnose from the hook-local step.
    include_ude_sections::Bool = true,
    # If non-empty, only these hook names are valid. Restricting the list
    # removes excluded positions instead of arguing against them in prose.
    allowed_hooks::Vector{Symbol} = Symbol[],
    # With true, each offered type is listed with its constitutive equations
    # (see `_component_law_lines`). Changes the prompt, so false by default.
    include_component_equations::Bool = false,
)::String
    io = IOBuffer()
    println(io, "You are assisting AIRMED, a digital-twin framework.")
    if include_ude_sections
        println(io, "A ModelingToolkit (MTK) base model has been augmented with a neural")
        println(io, "network correction (UDE). You need to suggest concrete physical")
        println(io, "components that, if added to the base model, would explain the")
        println(io, "correction the NN learned.")
    else
        println(io, "A drift was detected between a ModelingToolkit (MTK) base model and")
        println(io, "measured data, and re-fitting the existing parameters did not explain")
        println(io, "it. You need to suggest concrete physical components that, if added to")
        println(io, "the base model, would account for the measurements below.")
    end
    println(io)
    println(io, "## Problem")
    println(io, "Name: $(problem.name)")
    println(io, "Time span: $(problem.tspan)")
    println(io)

    println(io, "## Base model components")
    for sys in ModelingToolkit.get_systems(problem.model)
        println(io, "  - $(nameof(sys))")
    end
    println(io)

    println(io, "## Base model equations / connections")
    for eq in filter(_is_connect_eq, equations(problem.model))
        println(io, "  $(_eq_to_connect_display(eq))")
    end
    println(io)

    println(io, "## Base model parameters (known physics)")
    if isempty(problem.p0)
        println(io, "  (none)")
    else
        for (k, v) in problem.p0
            println(io, "  $(k) = $(v)")
        end
    end
    println(io)

    println(io, "## Initial conditions")
    if isempty(problem.u0)
        println(io, "  (none)")
    else
        for (k, v) in problem.u0
            println(io, "  $(k) = $(v)")
        end
    end
    println(io)

    if !isempty(problem.observable_states)
        println(io, "## Physically observed states (rows in measurement data)")
        n_ude_states = length(problem.u0)
        for s in problem.observable_states
            # ODESystem entries (e.g. subsystem ports) are not scalar sensor variables;
            # print only their name to avoid multi-line MTK model descriptions in the prompt.
            if s isa ModelingToolkit.AbstractSystem
                println(io, "  - $(nameof(s)) (subsystem — not a scalar sensor variable)")
            else
                println(io, "  - $s")
            end
        end
        if n_ude_states < length(problem.observable_states)
            n_extra = length(problem.observable_states) - n_ude_states
            if include_ude_sections
                println(io, "  NOTE: The UDE was trained on only $(n_ude_states) dynamic state(s). " *
                            "The remaining $(n_extra) " *
                            "sensor(s) above were NOT seen during NN training — the NN correction " *
                            "captures only what is visible from the $(n_ude_states) dynamic state(s). " *
                            "However, these extra sensors appear in the 'Sensor residuals' section " *
                            "below and carry structural information that the NN alone cannot provide.")
            else
                # The same state-versus-sensor distinction without naming a
                # UDE, which this prompt variant never introduces.
                println(io, "  NOTE: The model has $(n_ude_states) dynamic state(s). " *
                            "The remaining $(n_extra) sensor(s) above are algebraic consequences " *
                            "of those state(s) — they have no differential equation of their own. " *
                            "They appear in the 'Sensor residuals' section below, where they carry " *
                            "structural information about WHERE the mismatch originates that the " *
                            "dynamic state(s) alone cannot localise.")
            end
        end
        println(io)
    end

    if include_ude_sections
    println(io, "## Neural network correction (UDE) — architecture and interpretation")
    if !isnothing(update_result.nn)
        try
            n_params = length(ComponentArray(update_result.trained_nn_params))
            println(io, "  Trainable parameters: $n_params")
        catch
        end
        for (i, layer) in enumerate(update_result.nn.layers)
            println(io, "  [$i] $layer")
        end
    elseif !isnothing(update_result.sym_nn)
        println(io, "  (symbolic NN via @SymbolicNeuralNetwork — architecture encoded in sym_nn)")
    end
    println(io, "  Training loss: $(update_result.final_loss)  " *
                "($(update_result.n_iterations) iterations)")
    ex = get(update_result.metrics, :drift_explained, nothing)
    if ex isa AbstractVector && any(v -> !isnan(v), ex)
        pct = join([isnan(v) ? "n/a" : "$(round(100 * Float64(v); sigdigits=3))%"
                    for v in ex], ", ")
        println(io, "  Drift explained by the NN correction (per state): $pct")
        println(io, "  (fraction of the base-model residual the correction removed —")
        println(io, "   the higher it is, the more faithfully the characterisation below")
        println(io, "   reflects the missing physics)")
    end
    println(io)
    println(io, "  The UDE adds a learned scalar correction to each state derivative:")
    for (i, (k, v0)) in enumerate(problem.u0)
        println(io, "    d($k)/dt  =  [base model ODE]  +  NN_output[$i]")
    end
    println(io, "  A persistent non-zero correction indicates a structural mismatch between")
    println(io, "  the base model and the true system — a component is present in reality")
    println(io, "  that is absent from the base model.")
    println(io)

    println(io, "## IMPORTANT: Structural identifiability — use sensor residuals, not the UDE alone")
    println(io)
    println(io, "  The NN correction was trained on $(length(problem.u0)) dynamic state(s).")
    if length(problem.u0) == 1
        println(io, "  With a single observed dynamic state the UDE correction CANNOT distinguish")
        println(io, "  structural topologies that produce the same effective dynamics.")
        println(io, "  The additional sensor channels listed in 'Physically observed states' may")
        println(io, "  break this degeneracy by exposing measurements that depend differently on")
        println(io, "  series-inserted vs. parallel-inserted components.")
        println(io, "  Use the sensor residuals section below to decide WHICH hook positions")
        println(io, "  actually require a new component.")
    end
    println(io, "  Do NOT fill every hook automatically — propose a component only where")
    println(io, "  the sensor residuals provide physical evidence for it.")
    println(io)
    end  # include_ude_sections (architecture + identifiability note)

    if !include_ude_sections
        # The identifiability warning caveats the UDE; without it the restraint
        # instruction still has to be stated explicitly.
        println(io, "  Propose a component ONLY at positions the measurements below support.")
        println(io, "  Do NOT fill every hook automatically.")
        println(io)
    end

    if !isempty(sensor_summary)
        println(io, "## Sensor residuals (measured data − base model prediction)")
        println(io, "  Use these residuals to identify WHICH hook positions need new components.")
        println(io, "  They encode information that the UDE correction alone cannot provide.")
        println(io)
        println(io, sensor_summary)
        println(io)
    end

    # Placed after the sensor residuals and before the network
    # characterisation: the most specific evidence, with a fitted parameter.
    if !isempty(hook_summary)
        println(io, "## Hook-local characterisation (measured, not inferred)")
        println(io, "  This is the strongest evidence in this prompt. Where a component's")
        println(io, "  parameter is already identified below, do NOT re-derive it — decide")
        println(io, "  which physical component it corresponds to and whether that is")
        println(io, "  plausible for this system.")
        println(io)
        println(io, hook_summary)
        println(io)
    end

    if include_ude_sections
        println(io, "## NN correction characterisation (trained on dynamic state only)")
        println(io, nn_summary)
        println(io)
    end

    println(io, "## ComponentHook positions (where unknown components could be inserted)")
    println(io, "  Propose a component ONLY at positions where sensor residuals support it.")
    # `allowed_hooks` applies here too; restricting only the list of valid
    # values would leave this section describing every hook.
    for h in (isempty(allowed_hooks) ? problem.component_hooks :
              filter(h -> h.name in allowed_hooks, problem.component_hooks))
        a = "$(h.port_a[1]).$(h.port_a[2])"
        b = "$(h.port_b[1]).$(h.port_b[2])"
        topo = h.port_a[1] == h.port_b[1] ? "parallel with $(h.port_a[1])" :
                                              "series between $a and $b"
        println(io, "  - $(h.name): $a ↔ $b  ($topo)")
        println(io, "      $(h.description)")
    end
    println(io)

    println(io, "## Available component types ($(problem.component_library_label))")
    if include_component_equations
        # Without this the types are bare names, while the constitutive law
        # separates, say, constant power from a static impedance.
        println(io, "  Each type is listed with the equations it contributes, read from the")
        println(io, "  component itself. `v` is the voltage across its two pins, `i` the")
        println(io, "  current through it (v = p.v − n.v, i = p.i = −n.i), `t` is time.")
        println(io, "  Numeric coefficients shown are FIXED characteristics of the component;")
        println(io, "  the free parameter named on each line is fitted afterwards.")
    end
    for g in problem.component_guesses
        # `description` is often just the type name repeated ("- Consumer:
        # Consumer"); print it only when it actually says something more.
        println(io, strip(g.description) == string(g.component_type) ?
                    "  - $(g.component_type)" : "  - $(g.component_type): $(g.description)")
        include_component_equations || continue
        laws = _component_law_lines(g)
        println(io, "      free parameter: $(g.parameter_name)  (fitted numerically once the structure is fixed)")
        if isempty(laws)
            println(io, "      equations: (unavailable)")
        else
            println(io, "      equations:")
            for l in laws
                println(io, "        $l")
            end
        end
    end
    println(io)

    # Inject failed-attempt context on retries so the LLM can reason about
    # what structural choices were insufficient and propose alternatives.
    if !isempty(failed_attempts)
        println(io, "## Previous failed adaptation attempts")
        println(io, "  The proposals below were tried, parameters were fitted to measurement data,")
        println(io, "  and drift was still detected in the adapted model's residuals.")
        println(io, "  Study these failures carefully and propose a DIFFERENT structure.")
        println(io)
        for text in failed_attempts
            println(io, text)
        end
        println(io)
    end

    println(io, "## Task")
    println(io, "Return ONLY a JSON object (no markdown prose outside the JSON).")
    println(io, "The JSON must have exactly two top-level fields: \"proposals\" and \"julia_model_code\".")
    println(io)
    println(io, "Each entry in \"proposals\" must have exactly these four string fields:")
    println(io, "  - hook          : EXACT hook name from the ComponentHook list above")
    # Deliberately neutral: an example identifier such as "r2_series" would hint
    # at a component type AND a topology the model should be deriving itself.
    println(io, "  - name          : short Julia identifier for the new component")
    println(io, "  - component_type: one of $(join(["\"$(g.component_type)\"" for g in problem.component_guesses], ", "))")
    println(io, "  - rationale     : one sentence")
    println(io)
    offered = isempty(allowed_hooks) ? [h.name for h in problem.component_hooks] :
              [h.name for h in problem.component_hooks if h.name in allowed_hooks]
    println(io, "The \"hook\" field MUST be exactly one of:")
    for h in offered
        println(io, "  \"$(h)\"")
    end
    if !isempty(allowed_hooks) && length(offered) < length(problem.component_hooks)
        println(io, "  (Hook positions whose measured balance-law residual sits at the sensor")
        println(io, "   noise floor are NOT listed: the data excludes a missing component")
        println(io, "   there, so proposing one is not an option.)")
        length(offered) == 1 &&
            println(io, "  Exactly one position remains. Propose ONE component there.")
    end
    println(io, "Do NOT add port_a/port_b — they are derived from the hook name automatically.")
    println(io, "Do NOT add comments inside the JSON. Do NOT use backtick strings.")
    println(io)
    # Placeholders only: a real hook or type would anchor the answer, and an
    # unsubstituted placeholder matches no hook and is discarded.
    if !isempty(problem.component_hooks)
        println(io, "Format example — the values below are PLACEHOLDERS showing the field")
        println(io, "names and value types ONLY. They are not suggestions: replace each")
        println(io, "<...> with an actual choice from the lists above.")
        println(io, """```json
{
  "proposals": [
    {
      "hook": "<exact hook name from the list above>",
      "name": "<short julia identifier of your choice>",
      "component_type": "<exact component type from the list above>",
      "rationale": "<one sentence citing the specific evidence for this choice>"
    }
  ],
  "julia_model_code": ""
}
```""")
    end
    println(io)
    println(io, "IMPORTANT: Do NOT include parameter values — they are identified by")
    println(io, "numerical optimisation after the structural proposal is fixed.")
    println(io, "Propose components ONLY where the sensor residuals or NN characterisation")
    println(io, "provide physical evidence. You do NOT need to fill every hook position.")
    println(io, "Fewer, well-justified proposals are better than filling every available slot.")
    return String(take!(io))
end

# --- Internals: LLM dispatch ------------------------------------------------

"""
    _call_llm_text(api, prompt, api_key; base_url, model, timeout, log_file) -> String

Free-text LLM call for an explanation rather than a proposal, reusing the
backend dispatch of `_call_llm`. Used so that an escalation is explained by the
backend that produced the rejected proposal, unlike `explain_for_user`, which
uses the one in `SupervisionConfig`.

# Arguments
- `api`, `api_key`: the backend and its key.
- `prompt`: the question to answer.

# Keywords
- `base_url`, `model`: endpoint and model name.
- `num_ctx`, `think`: `:ollama_native` only.
- `timeout`: seconds per call.
- `log_file`: file to append the exchange to.

# Returns
The raw response text.
"""
function _call_llm_text(api::Symbol, prompt::AbstractString, api_key::AbstractString;
                        base_url::AbstractString = "", model::AbstractString = "",
                        num_ctx::Union{Nothing, Integer} = nothing,
                        think::Union{Nothing, Bool}      = nothing,
                        timeout::Real = 180,
                        log_file::Union{Nothing, AbstractString} = nothing)::String
    api === :none && return "[no LLM configured — set api on propose_model_adaptation " *
                            "to enable a plain-language explanation]"
    raw_text, _, _ = _call_llm(api, prompt, api_key; base_url, model, num_ctx, think, timeout, log_file)
    return raw_text
end

function _call_llm(api::Symbol, prompt::AbstractString, api_key::AbstractString;
                   base_url::AbstractString = "", model::AbstractString = "",
                   hooks::Vector{ComponentHook}   = ComponentHook[],
                   aliases::Dict{String, String} = Dict{String, String}(),
                   default_component_type::Union{Nothing, Symbol} = nothing,
                   # `:ollama_native` only; other backends ignore them and
                   # `nothing` keeps their existing behaviour.
                   num_ctx::Union{Nothing, Integer} = nothing,
                   think::Union{Nothing, Bool}      = nothing,
                   log_file::Union{Nothing, AbstractString} = nothing,
                   timeout::Real = 180)
    if api === :none
        return ("[demo mode — no LLM was called]", ComponentProposal[], "")
    end
    raw_text, proposals, code = if api === :anthropic
        _call_anthropic(prompt, api_key; model, hooks, aliases, default_component_type, timeout)
    elseif api === :ollama_native
        # The native endpoint, required for reasoning models: only /api/chat
        # honours `think`. A separate symbol keeps `:ollama` on /v1.
        _call_ollama_native(prompt; base_url, model, hooks, aliases,
                            default_component_type, num_ctx, think, timeout)
    elseif api === :openai || api === :ollama || api === :groq || api === :openrouter
        # All of these speak the OpenAI-compatible chat/completions format.
        # The actual server is selected by base_url; api is just a label.
        _call_openai(prompt, api_key; base_url, model, hooks, aliases, default_component_type, timeout)
    elseif api === :claude_cli
        # A local Claude Code CLI in print mode instead of an HTTP endpoint;
        # `base_url` carries the executable path, else AIRMED_CLAUDE_CLI.
        _call_claude_cli(prompt; model, cli_path = base_url,
                         hooks, aliases, default_component_type, timeout)
    else
        @warn "Unknown LLM api $api — falling back to demo mode."
        return ("[unknown api $api — no LLM was called]", ComponentProposal[], "")
    end
    _write_llm_log(log_file, api, model, base_url, prompt, raw_text, proposals)
    return raw_text, proposals, code
end

function _call_anthropic(prompt::AbstractString, api_key::AbstractString;
                          model::AbstractString = "",
                          hooks::Vector{ComponentHook} = ComponentHook[],
                          aliases::Dict{String, String} = Dict{String, String}(),
                          default_component_type::Union{Nothing, Symbol} = nothing,
                          timeout::Real = 180)
    if isempty(api_key)
        @warn "No Anthropic API key supplied — returning stub response."
        return ("[anthropic stub — no api_key]", ComponentProposal[], "")
    end
    try
        url  = "https://api.anthropic.com/v1/messages"
        hdrs = [
            "x-api-key"         => api_key,
            "anthropic-version" => "2023-06-01",
            "content-type"      => "application/json",
        ]
        model_name = isempty(model) ? "claude-opus-4-8" : String(model)
        # Forced tool use: the answer must be a `propose_components` call
        # matching this schema, so no JSON repair is needed.
        tool = Dict(
            "name"        => "propose_components",
            "description" => "Submit structural component proposals for the adapted model.",
            "input_schema" => Dict(
                "type"       => "object",
                "properties" => Dict(
                    "proposals" => Dict(
                        "type"  => "array",
                        "items" => Dict(
                            "type"       => "object",
                            "properties" => Dict(
                                "hook"           => Dict("type" => "string",
                                                          "description" => "Exact ComponentHook name from the prompt"),
                                "name"           => Dict("type" => "string",
                                                          "description" => "Short Julia identifier for the new component"),
                                "component_type" => Dict("type" => "string",
                                                          "description" => "One of the listed component types"),
                                "rationale"      => Dict("type" => "string"),
                            ),
                            "required" => ["hook", "name", "component_type"],
                        ),
                    ),
                    "julia_model_code" => Dict("type" => "string"),
                ),
                "required" => ["proposals"],
            ),
        )
        body = JSON3.write(Dict(
            "model"       => model_name,
            "max_tokens"  => 4096,
            "tools"       => [tool],
            "tool_choice" => Dict("type" => "tool", "name" => "propose_components"),
            "messages"    => [Dict("role" => "user", "content" => String(prompt))],
        ))
        resp = HTTP.post(url, hdrs, body;
                         connect_timeout = 10, readtimeout = round(Int, timeout), retry = true, retries = 2)
        raw  = String(resp.body)
        obj  = JSON3.read(raw)

        tool_input = nothing
        text_parts = String[]
        for block in get(obj, "content", [])
            btype = get(block, "type", "")
            if btype == "tool_use" && get(block, "name", "") == "propose_components"
                tool_input = block["input"]
            elseif btype == "text"
                push!(text_parts, String(block["text"]))
            end
        end

        if !isnothing(tool_input)
            proposals, code = _proposals_from_json_obj(tool_input, hooks;
                                                       aliases, default_component_type)
            return JSON3.write(tool_input), proposals, code
        end
        # Fallback for a response without a tool call: parse the text blocks.
        text = join(text_parts, "\n")
        proposals, code = _parse_llm_proposals(text, hooks;
                                                aliases, default_component_type)
        return text, proposals, code
    catch err
        @warn "Anthropic API call failed: $(sprint(showerror, err))"
        return ("[anthropic error: $(sprint(showerror, err))]", ComponentProposal[], "")
    end
end

"""
    _call_ollama_native(prompt; base_url, model, num_ctx, think, ...) -> (raw_text, proposals, code)

The native `/api/chat` endpoint of Ollama, used instead of its
OpenAI-compatible shim for models that reason before answering. Only this
endpoint honours `think` and `options.num_ctx`; the shim ignores the first and
allocates the declared maximum context.

# Arguments
- `prompt`: the prompt to send.

# Keywords
- `base_url`, `model`: endpoint and model name.
- `num_ctx`: context size. `nothing` uses `AIRMED_OLLAMA_NUM_CTX`.
- `think`: reasoning pass. `nothing` uses `AIRMED_OLLAMA_THINK`.
- `hooks`, `aliases`, `default_component_type`: passed to the parser.
- `timeout`: seconds before the request is abandoned.

# Returns
`(raw_text, proposals, code)`.
"""
function _call_ollama_native(prompt::AbstractString;
                             base_url::AbstractString = "",
                             model::AbstractString    = "",
                             hooks::Vector{ComponentHook}  = ComponentHook[],
                             aliases::Dict{String, String} = Dict{String, String}(),
                             default_component_type::Union{Nothing, Symbol} = nothing,
                             num_ctx::Union{Nothing, Integer} = nothing,
                             think::Union{Nothing, Bool}      = nothing,
                             timeout::Real = 600)
    # Callers configure the /v1 shim; the native API lives at the server root.
    root = rstrip(isempty(base_url) ? "http://localhost:11434" : base_url, '/')
    endswith(root, "/v1") && (root = root[1:end-3])
    url  = rstrip(root, '/') * "/api/chat"
    # `nothing` (the default) preserves the original env-var-only behaviour;
    # an explicit value (e.g. threaded from SupervisionConfig) overrides it.
    ctx_val   = something(num_ctx, parse(Int, get(ENV, "AIRMED_OLLAMA_NUM_CTX", "8192")))
    think_val = something(think,   get(ENV, "AIRMED_OLLAMA_THINK", "0") == "1")
    try
        body = JSON3.write(Dict(
            "model"    => isempty(model) ? "llama3.1" : String(model),
            "stream"   => false,
            # `think = false` by default: reasoning shares the num_predict
            # allowance, and an exhausted budget returns empty `content`.
            "think"    => think_val,
            "options"  => Dict(
                "num_ctx"     => ctx_val,
                "num_predict" => parse(Int, get(ENV, "AIRMED_LLM_MAX_TOKENS", "2048")),
            ),
            "messages" => [Dict("role" => "user", "content" => String(prompt))],
        ))
        resp = HTTP.post(url, ["content-type" => "application/json"], body;
                         connect_timeout = 10, readtimeout = round(Int, timeout),
                         retry = true, retries = 2)
        obj  = JSON3.read(String(resp.body))
        msg  = get(obj, "message", Dict())
        text = String(get(msg, "content", ""))
        get(obj, "done_reason", "stop") == "length" &&
            @warn "Ollama response hit num_predict — raise AIRMED_LLM_MAX_TOKENS."
        think_len = length(String(get(msg, "thinking", "")))
        isempty(strip(text)) &&
            @warn "Ollama returned empty content while emitting $(think_len) chars of " *
                  "reasoning — `think = false` is not being honoured by this server/model."
        proposals, code = _parse_llm_proposals(text, hooks;
                                               aliases, default_component_type)
        return text, proposals, code
    catch err
        @warn "Ollama native API call failed ($url): $(sprint(showerror, err))"
        return ("[ollama error: $(sprint(showerror, err))]", ComponentProposal[], "")
    end
end

function _call_openai(prompt::AbstractString, api_key::AbstractString;
                     base_url::AbstractString = "",
                     model::AbstractString    = "",
                     hooks::Vector{ComponentHook}    = ComponentHook[],
                     aliases::Dict{String, String} = Dict{String, String}(),
                     default_component_type::Union{Nothing, Symbol} = nothing,
                     timeout::Real = 180)
    # Require an API key only for the real OpenAI cloud endpoint; local servers
    # (Ollama, LM Studio, etc.) use an empty key with a custom base_url.
    if isempty(api_key) && isempty(base_url)
        @warn "No OpenAI API key supplied (and no base_url to override the endpoint) — returning stub response."
        return ("[openai stub — no api_key]", ComponentProposal[], "")
    end
    url        = isempty(base_url) ? "https://api.openai.com/v1/chat/completions" : rstrip(base_url, '/') * "/chat/completions"
    model_name = isempty(model) ? "gpt-4o" : model
    hdrs = isempty(api_key) ? ["content-type" => "application/json"] :
                              ["Authorization" => "Bearer $api_key", "content-type" => "application/json"]
    try
        # Reasoning models spend part of the budget before emitting `content`,
        # and a truncated answer is recorded as unparseable.
        body = JSON3.write(Dict(
            "model"      => model_name,
            "max_tokens" => parse(Int, get(ENV, "AIRMED_LLM_MAX_TOKENS", "2048")),
            "messages"   => [Dict("role" => "user", "content" => String(prompt))],
        ))
        resp = HTTP.post(url, hdrs, body;
                         connect_timeout = 10, readtimeout = round(Int, timeout), retry = true, retries = 2)
        raw  = String(resp.body)
        obj  = JSON3.read(raw)
        finish = get(get(get(obj, "choices", [Dict()])[1], "message", Dict()),
                     "finish_reason", "unknown")
        finish == "length" && @warn "LLM response truncated (finish_reason=length) — " *
                                    "increase max_tokens or shorten the prompt."
        text = String(obj["choices"][1]["message"]["content"])
        proposals, code = _parse_llm_proposals(text, hooks;
                                                aliases, default_component_type)
        return text, proposals, code
    catch err
        @warn "OpenAI-compatible API call failed ($url): $(sprint(showerror, err))"
        return ("[openai error: $(sprint(showerror, err))]", ComponentProposal[], "")
    end
end

# Tools denied to the Claude Code CLI backend: the prompt is self-contained,
# so file, command and web access would only add context it does not carry.
const CLAUDE_CLI_DISALLOWED_TOOLS = "Bash,BashOutput,KillShell,Read,Write,Edit," *
    "NotebookEdit,Glob,Grep,WebFetch,WebSearch,Task,Agent,TodoWrite,Skill"

# Flags that reduce a `claude -p` session to "answer this prompt, nothing else".
# See `_call_claude_cli`'s docstring for what each one closes off.
const CLAUDE_CLI_ISOLATION_ARGS = String[
    "--safe-mode",
    "--strict-mcp-config",
    "--disallowed-tools", CLAUDE_CLI_DISALLOWED_TOOLS,
]

"""
    _call_claude_cli(prompt; model, cli_path, workdir, isolation_args, hooks,
                     aliases, default_component_type, timeout)
        -> (raw_text, proposals, code)

LLM backend that runs a local Claude Code CLI in print mode, with the prompt
on stdin, and parses its stdout like any other free-text backend.

A CLI invocation is an agent session, so it would otherwise add context the
framework did not author: CLAUDE.md files, memory, settings, skills, plugins,
hooks, MCP servers and file and web tools. The isolation flags and the neutral
working directory suppress all of it, leaving the call equivalent to the HTTP
backends.

# Keywords
- `model`: passed as `--model`; an alias or a full id, omitted when empty.
- `cli_path`: executable path; empty uses `AIRMED_CLAUDE_CLI`, else `PATH`.
- `workdir`: working directory, a temporary one by default, which bounds file
  access and memory discovery.
- `isolation_args`: the isolation flags; override only if a CLI version
  rejects one.
- `timeout`: seconds before the child is killed.

# Returns
`(raw_text, proposals, code)`. On failure the text starts with
`[claude_cli error`, and stderr is surfaced only then.
"""
function _call_claude_cli(prompt::AbstractString;
                          model::AbstractString    = "",
                          cli_path::AbstractString  = "",
                          workdir::AbstractString   = tempdir(),
                          isolation_args::Vector{String} = CLAUDE_CLI_ISOLATION_ARGS,
                          hooks::Vector{ComponentHook}    = ComponentHook[],
                          aliases::Dict{String, String} = Dict{String, String}(),
                          default_component_type::Union{Nothing, Symbol} = nothing,
                          timeout::Real = 180)
    exe = !isempty(cli_path) ? cli_path : get(ENV, "AIRMED_CLAUDE_CLI", "claude")
    args = String["-p"]
    append!(args, isolation_args)
    isempty(model) || append!(args, ["--model", String(model)])
    out = IOBuffer(); err = IOBuffer()
    timedout = Ref(false)
    try
        cmd  = Cmd(`$exe $args`; dir = workdir)
        proc = run(pipeline(cmd; stdin = IOBuffer(String(prompt)),
                            stdout = out, stderr = err); wait = false)
        # readtimeout-style watchdog: fires once; kills the child if it overruns.
        timer = Timer(timeout) do _
            if process_running(proc)
                timedout[] = true
                kill(proc)
            end
        end
        try
            wait(proc)
        finally
            close(timer)
        end
        timedout[] && return ("[claude_cli error: timeout after $(round(Int, timeout))s]",
                              ComponentProposal[], "")
        if !success(proc)
            errtxt = strip(String(take!(err)))
            return ("[claude_cli error: exit $(proc.exitcode) — $(first(errtxt, 300))]",
                    ComponentProposal[], "")
        end
        text = String(take!(out))
        proposals, code = _parse_llm_proposals(text, hooks; aliases, default_component_type)
        return text, proposals, code
    catch e
        @warn "Claude CLI call failed ($exe): $(sprint(showerror, e))"
        return ("[claude_cli error: $(sprint(showerror, e))]", ComponentProposal[], "")
    end
end

"""
    _write_llm_log(path, api, model, base_url, prompt, response, proposals)

Append one call record to `path`, delimited by a header line so the file stays
greppable. Does nothing when `path` is `nothing`.

# Arguments
- `path`: log file, or `nothing`.
- `api`, `model`, `base_url`: which backend was called.
- `prompt`, `response`: the exchange.
- `proposals`: what was parsed from the response.

# Returns
`nothing`; the record is appended to the file.
"""
function _write_llm_log(
    path::Union{Nothing, AbstractString},
    api::Symbol,
    model::AbstractString,
    base_url::AbstractString,
    prompt::AbstractString,
    response::AbstractString,
    proposals::Vector{ComponentProposal},
)
    isnothing(path) && return
    try
        open(path, "a") do io
            sep = "=" ^ 72
            println(io, sep)
            println(io, "AIRMED LLM CALL  $(Dates.now())")
            endpoint = isempty(base_url) ? string(api) : "$api  $base_url"
            isempty(model) || (endpoint *= "  model=$model")
            println(io, "Endpoint: $endpoint")
            println(io, "Proposals parsed: $(length(proposals))")
            for p in proposals
                pstr = join(["$k=$(round(v; sigdigits=4))" for (k,v) in p.parameters], ", ")
                println(io, "  $(p.name) :: $(p.component_type)  [$pstr]")
            end
            println(io, "--- PROMPT ---")
            println(io, prompt)
            println(io, "--- RESPONSE ---")
            println(io, response)
            println(io, sep)
            println(io)
        end
    catch err
        @warn "LLM log write failed ($(repr(path))): $(sprint(showerror, err))"
    end
end

# Normalise port names with the caller's `port_aliases`; AIRMED defines no
# domain-specific aliases itself.
_normalise_port(s::AbstractString, aliases::Dict{String, String}) =
    get(aliases, s, s)

# Parse the returned JSON into ComponentProposal objects. With `hooks`, a
# proposal may name a hook instead of explicit ports, which are then resolved.
function _parse_llm_proposals(text::AbstractString,
                               hooks::Vector{ComponentHook} = ComponentHook[];
                               aliases::Dict{String, String} = Dict{String, String}(),
                               default_component_type::Union{Nothing, Symbol} = nothing)
    json_str = _extract_json_block(text)
    if isnothing(json_str)
        @warn "LLM response contained no parseable JSON — using demo-mode fallback."
        return ComponentProposal[], ""
    end
    obj = try
        JSON3.read(json_str)
    catch err
        @warn "Failed to parse LLM JSON proposals: $(sprint(showerror, err))"
        return ComponentProposal[], ""
    end
    return _proposals_from_json_obj(obj, hooks; aliases, default_component_type)
end

# Convert an already-parsed JSON object (from free-text extraction or from a
# structured tool-use response) into ComponentProposal objects + optional code.
function _proposals_from_json_obj(obj,
                                  hooks::Vector{ComponentHook} = ComponentHook[];
                                  aliases::Dict{String, String} = Dict{String, String}(),
                                  default_component_type::Union{Nothing, Symbol} = nothing)
    hook_by_name = Dict(string(h.name) => h for h in hooks)
    try
        proposals = ComponentProposal[]
        for p in get(obj, "proposals", [])
            try
                hook_name = get(p, "hook", nothing)
                hook_name_s = isnothing(hook_name) ? nothing : String(hook_name)
                resolved_hook = isnothing(hook_name_s) ? nothing :
                                get(hook_by_name, hook_name_s, nothing)

                # Default component type when the field is omitted: the
                # caller's first ComponentGuess, else :Unknown.
                ct_default = isnothing(default_component_type) ? :Unknown :
                                                                  default_component_type

                if !isnothing(resolved_hook)
                    # Hook-name schema: ports come from the ComponentHook definition.
                    push!(proposals, ComponentProposal(
                        Symbol(get(p, "name",           hook_name_s)),
                        Symbol(get(p, "component_type", ct_default)),
                        resolved_hook.port_a,
                        resolved_hook.port_b,
                        Dict{Symbol, Float64}(),
                        String(get(p, "rationale", "")),
                    ))
                elseif !isnothing(hook_name_s)
                    @warn "Skipping proposal '$(get(p,"name","?"))': hook \"$hook_name_s\" not found. " *
                          "Valid hook names: $(join(keys(hook_by_name), ", "))"
                else
                    # Legacy schema: explicit port_a/port_b arrays.
                    pa = get(p, "port_a", ["", ""])
                    pb = get(p, "port_b", ["", ""])
                    length(pa) >= 2 || (@warn "Skipping proposal '$(get(p,"name","?"))': port_a needs [\"subsystem\",\"port\"], got $(pa)"; continue)
                    length(pb) >= 2 || (@warn "Skipping proposal '$(get(p,"name","?"))': port_b needs [\"subsystem\",\"port\"], got $(pb)"; continue)
                    pa_port = _normalise_port(String(pa[2]), aliases)
                    pb_port = _normalise_port(String(pb[2]), aliases)
                    push!(proposals, ComponentProposal(
                        Symbol(get(p, "name",           "unknown")),
                        Symbol(get(p, "component_type", ct_default)),
                        (Symbol(pa[1]), Symbol(pa_port)),
                        (Symbol(pb[1]), Symbol(pb_port)),
                        Dict{Symbol, Float64}(),
                        String(get(p, "rationale", "")),
                    ))
                end
            catch err2
                @warn "Skipping malformed proposal: $(sprint(showerror, err2))" maxlog=5
            end
        end
        code = String(get(obj, "julia_model_code", ""))
        return proposals, code
    catch err
        @warn "Failed to parse LLM JSON proposals: $(sprint(showerror, err))"
        return ComponentProposal[], ""
    end
end

# Strip `//` and `#` comments, which models add although JSON forbids them.
# Byte-safe iteration keeps multi-byte characters from raising StringIndexError.
function _strip_json_comments(s::AbstractString)::String
    buf = IOBuffer()
    for line in eachline(IOBuffer(String(s)))
        in_str  = false
        escaped = false
        cut_at  = nothing          # byte index of the comment-start character
        pos     = firstindex(line)
        while pos <= lastindex(line)
            c = line[pos]
            if escaped
                escaped = false
                pos = nextind(line, pos)
                continue
            end
            if c == '\\' && in_str
                escaped = true
                pos = nextind(line, pos)
                continue
            end
            if c == '"'
                in_str = !in_str
                pos = nextind(line, pos)
                continue
            end
            if !in_str
                if c == '#'
                    cut_at = pos
                    break
                end
                if c == '/'
                    nxt = nextind(line, pos)
                    if nxt <= lastindex(line) && line[nxt] == '/'
                        cut_at = pos
                        break
                    end
                end
            end
            pos = nextind(line, pos)
        end
        println(buf, isnothing(cut_at) ? line : line[1:prevind(line, cut_at)])
    end
    return String(take!(buf))
end

"""
    _parse_escalation_json(raw_text) -> (explanation::String, recommendation::String, ok::Bool)

Parse an escalation explanation into its two fields, reusing
`_extract_json_block` for fenced, bare and truncated JSON.

# Arguments
- `raw_text`: the response to parse.

# Returns
`(explanation, recommendation, ok)`. `ok` is `false` on missing JSON, a
non-string field or an empty string, and the caller then keeps the raw text.
"""
function _parse_escalation_json(raw_text::AbstractString)
    json_str = _extract_json_block(raw_text)
    isnothing(json_str) && return ("", "", false)
    obj = try
        JSON3.read(json_str)
    catch
        return ("", "", false)
    end
    expl = get(obj, "explanation", nothing)
    rec  = get(obj, "recommendation", nothing)
    (isnothing(expl) || isnothing(rec)) && return ("", "", false)
    expl_s, rec_s = String(expl), String(rec)
    (isempty(strip(expl_s)) || isempty(strip(rec_s))) && return ("", "", false)
    return (expl_s, rec_s, true)
end

# Render (explanation, recommendation) as JSON, field by field so the output
# stays readable while `JSON3.write` escapes each value.
_escalation_to_json(explanation::AbstractString, recommendation::AbstractString) =
    "{\n  \"explanation\": " * JSON3.write(String(explanation)) * ",\n" *
    "  \"recommendation\": " * JSON3.write(String(recommendation)) * "\n}"

# Extract the first JSON object, preferring a fenced block and falling back to
# brace counting. Byte-safe iteration avoids StringIndexError.
function _extract_json_block(text::AbstractString)
    text = _strip_json_comments(text)
    # Normalise multi-line string syntax that is invalid in JSON:
    # Julia-style triple-quoted strings """..."""
    text = replace(text, r"\"\"\".*?\"\"\""s => "\"\"")
    # Backtick-delimited strings `...` (JavaScript/Julia template literal syntax).
    # LLMs sometimes use backtick strings for the julia_model_code field.
    text = replace(text, r":\s*`[^`]*`"s => ": \"\"")
    # Prefer content from a fenced code block; the fence regex no longer requires a
    # complete {…} so that truncated responses (missing closing brace) are still caught.
    m = match(r"```(?:json)?\s*(.*?)\s*```"s, text)
    json_candidate = isnothing(m) ? text : String(m.captures[1])
    # Brace-count to extract the outermost object; repair truncated JSON if needed.
    result = _brace_extract(json_candidate)
    # If the fenced content had no '{', fall back to searching the full text.
    if isnothing(result) && !isnothing(m)
        result = _brace_extract(text)
    end
    return result
end

# Find the outermost balanced {...} in `s`.  If the object is truncated (depth > 0
# at end of string), append the missing closing braces and return the repaired string.
function _brace_extract(s::AbstractString)
    start = findfirst('{', s)
    isnothing(start) && return nothing
    depth = 0
    pos   = start
    while pos <= lastindex(s)
        c = s[pos]
        c == '{' && (depth += 1)
        c == '}' && (depth -= 1)
        depth == 0 && return SubString(s, start, pos)
        pos = nextind(s, pos)
    end
    # Truncated response: close unclosed objects by appending missing braces.
    depth > 0 && return s[start:lastindex(s)] * "}" ^ depth
    return nothing  # depth < 0: extra closing braces — malformed
end

# --- Internals: from ComponentGuess to ComponentProposal --------------------

"""
    _proposals_from_guesses(hooks, guesses; attempt=1) -> Vector{ComponentProposal}

Demo-mode fallback without an LLM: pair hooks with guesses into concrete
proposals.

# Arguments
- `hooks`: the positions to fill.
- `guesses`: the candidate component types.

# Keywords
- `attempt`: index into an enumeration ordered by increasing structure size, so
  every one-hook combination is tried before any two-hook one, and a single
  fault is found early whichever hook it sits at. There are
  `(n_guesses+1)^n_hooks - 1` combinations in total.

# Returns
The proposals of that combination, empty once the enumeration is exhausted.
"""
function _proposals_from_guesses(
    hooks::Vector{ComponentHook},
    guesses::Vector{ComponentGuess};
    attempt::Int = 1,
)::Vector{ComponentProposal}
    (isempty(hooks) || isempty(guesses)) && return ComponentProposal[]
    n_h, n_g = length(hooks), length(guesses)

    hook_idxs, type_idxs = _combo_at_rank(n_h, n_g, attempt)
    isnothing(hook_idxs) && return ComponentProposal[]   # rank beyond all combinations

    proposals = ComponentProposal[]
    for (h_idx, t_idx) in zip(hook_idxs, type_idxs)
        hook = hooks[h_idx]
        g    = guesses[t_idx]
        type_initial = lowercase(string(g.component_type)[1:1])
        push!(proposals, ComponentProposal(
            Symbol("$(type_initial)_$(hook.name)"),
            g.component_type,
            hook.port_a,
            hook.port_b,
            Dict{Symbol, Float64}(),
            "Demo mode attempt $attempt: $(g.component_type) inserted between " *
            "$(hook.port_a[1]).$(hook.port_a[2]) and $(hook.port_b[1]).$(hook.port_b[2])",
        ))
    end
    return proposals
end

# All k-element subsets of 1:n_h, as index vectors, via bitmask enumeration
# (no external combinatorics dependency needed for the small n_h this targets).
function _hook_subsets(n_h::Int, k::Int)::Vector{Vector{Int}}
    result = Vector{Int}[]
    for mask in 0:(2^n_h - 1)
        idxs = [i for i in 1:n_h if (mask >> (i - 1)) & 1 == 1]
        length(idxs) == k && push!(result, idxs)
    end
    return result
end

# The (hook indices, type indices) pair at position `attempt` of the
# size-ordered enumeration; (nothing, nothing) once it is exhausted.
function _combo_at_rank(n_h::Int, n_g::Int, attempt::Int)
    remaining = attempt
    for k in 1:n_h
        subsets   = _hook_subsets(n_h, k)
        n_subsets = length(subsets)
        n_level   = n_subsets * n_g^k
        if remaining <= n_level
            # Every hook subset is tried for one type assignment before the
            # next, so the most plausible type reaches all hooks first.
            type_rank  = (remaining - 1) ÷ n_subsets
            subset_idx = (remaining - 1) % n_subsets + 1
            type_idxs  = digits(type_rank; base = n_g, pad = k) .+ 1
            return subsets[subset_idx], type_idxs
        end
        remaining -= n_level
    end
    return nothing, nothing
end

# Stable fingerprint for a proposal set, used to detect duplicates across retry
# attempts so that the same structure is not evaluated twice.
_proposal_fingerprint(proposals::Vector{ComponentProposal}) =
    sort([(string(p.component_type), minmax(p.port_a, p.port_b)) for p in proposals])

# Format one failed adaptation attempt as a human-readable string for the LLM prompt.
function _format_failed_attempt(
    attempt_idx::Int,
    proposals::Vector{ComponentProposal},
    fit_loss::Float64,
    ude_loss::Float64,
    ratio::Float64,
    retry_ratio_threshold::Float64,
    drift_result::DriftResult,
    per_sensor::Vector{Pair{String, Float64}},
    bound_flags::Vector{String} = String[],
)::String
    io = IOBuffer()
    println(io, "Attempt $attempt_idx (FAILED — drift still detected in adapted model after fitting):")
    println(io, "  Proposed components:")
    for p in proposals
        topo = p.port_a[1] == p.port_b[1] ?
               "parallel with $(p.port_a[1])" :
               "series between $(p.port_a[1]).$(p.port_a[2]) and $(p.port_b[1]).$(p.port_b[2])"
        pstr = join(["$(k)=$(round(v; sigdigits=4))" for (k, v) in p.parameters], ", ")
        println(io, "    - $(p.name) :: $(p.component_type)  ($topo)" *
                    (isempty(pstr) ? "" : "  fitted: $pstr"))
    end
    if !isempty(per_sensor)
        println(io, "  Post-fit RMSE per sensor:")
        for (s, rmse) in per_sensor
            println(io, "    $s: $(round(rmse; sigdigits=4))")
        end
    end
    ratio_str = isnan(ratio) ? "N/A" : "$(round(ratio; sigdigits=3))×"
    println(io, "  Fit/UDE ratio: $ratio_str  (retry threshold: $(retry_ratio_threshold)×)")
    if drift_result.drift_detected
        idx_str = isnothing(drift_result.drift_index) ? "?" : string(drift_result.drift_index)
        println(io, "  Drift still detected at sample $idx_str " *
                    "(mean residual = $(round(drift_result.mean_residual; sigdigits=4)), " *
                    "max = $(round(drift_result.max_residual; sigdigits=4)))")
    end
    if !isempty(bound_flags)
        println(io, "  Physically implausible fitted parameters (pinned at plausibility bounds):")
        for f in bound_flags
            println(io, "    - $f")
        end
        println(io, "    A parameter at its bound means the optimiser ran away trying to make")
        println(io, "    this component type vanish or dominate — strong evidence the component")
        println(io, "    TYPE at that position is wrong.")
    end
    println(io, "  → INSTRUCTION: The structure above was insufficient. " *
                "Propose a DIFFERENT combination of component types and/or hook positions.")
    return String(take!(io))
end

# --- Internals: parameter fitting -------------------------------------------

# The matching ComponentGuess for a proposal, or `nothing` if no guess of that
# type is registered, which the caller reports in the generated code.
function _matching_guess(prop::ComponentProposal,
                          guesses::Vector{ComponentGuess})::Union{Nothing, ComponentGuess}
    idx = findfirst(g -> g.component_type == prop.component_type, guesses)
    return isnothing(idx) ? nothing : guesses[idx]
end

# Primary fittable parameter name, taken from the caller's ComponentGuess.
# Falls back to `:p` for proposals whose type has no registered guess.
function _main_param_name(prop::ComponentProposal,
                           guesses::Vector{ComponentGuess})::Symbol
    g = _matching_guess(prop, guesses)
    return isnothing(g) ? :p : g.parameter_name
end

# Default initial guess for the primary parameter, taken from the caller's
# ComponentGuess.  Falls back to 1.0 when no matching guess is registered.
function _default_param_val(prop::ComponentProposal,
                             guesses::Vector{ComponentGuess})::Float64
    g = _matching_guess(prop, guesses)
    return isnothing(g) ? 1.0 : g.default_parameter
end

"""
    _is_connect_eq(eq) -> Bool

`true` when `eq` is an explicit `connect(...)` equation. The algebraic
equations implied by a connection give `false` and are excluded from the
generated code and the prompt.
"""
function _is_connect_eq(eq)::Bool
    try
        # In MTK v9+ connect equations wrap a ModelingToolkit.Connection RHS.
        return isa(ModelingToolkit.value(eq.rhs), ModelingToolkit.Connection)
    catch
        # Fallback based on the string representation: connect equations
        # stringify as "connect(...)", expanded pin equations do not.
        return occursin("connect", string(eq))
    end
end

"""
    _build_adapted_ode_system(proposals, init_vals, problem)
        -> (ODESystem | Nothing, Vector{Pair{Int,Any}}, String)

Build the adapted ODESystem from the equation objects of the base model,
creating components through the `factory` of the matching `ComponentGuess`.

The same subsystem in both ports inserts in parallel, adding two `connect`
equations; two different ones insert in series, replacing the existing
connection.

# Returns
`(sys, param_syms, msg)`, with `sys` the simplified system or `nothing`, and
`param_syms` mapping each proposal index to its symbolic parameter.
"""
function _assemble_adapted_system(
    proposals::Vector{ComponentProposal},
    init_vals::Vector{Float64},
    problem::AIRMEDProblem,
)::Tuple{Union{ODESystem, Nothing}, Vector{Pair{Int,Any}}, String, Dict{Any,Any}, Union{Vector{ODESystem}, Nothing}}
    try
        # Connect equations only: the subsystem physics comes from the
        # ODESystem objects, and duplicates would over-constrain the system.
        base_eqs      = filter(_is_connect_eq, collect(equations(problem.model)))
        base_subsys   = ModelingToolkit.get_systems(problem.model)
        sys_dict      = Dict(nameof(s) => s for s in base_subsys)
        t_iv          = ModelingToolkit.get_iv(problem.model)
        new_components = ODESystem[]
        mod_eqs        = copy(base_eqs)

        # Symbolic parameters come from the standalone components, giving the
        # unscoped symbols `combined` uses; scoped ones break ODEProblem.
        comp_param_syms = Pair{Int, Any}[]

        # Initial connector values from ComponentGuess.connector_guesses, which
        # resolve cyclic symbolic substitutions during MTK initialisation.
        comp_guesses = Dict{Any, Any}()

        for (i, (prop, init_val)) in enumerate(zip(proposals, init_vals))
            # Locate the developer-supplied factory for this component type.
            g_idx = findfirst(g -> g.component_type == prop.component_type,
                              problem.component_guesses)
            isnothing(g_idx) &&
                return nothing, Pair{Int,Any}[], "no ComponentGuess factory for type $(prop.component_type)", Dict{Any,Any}(), nothing
            guess = problem.component_guesses[g_idx]
            comp = try
                guess.factory(prop.name, init_val)
            catch e
                return nothing, Pair{Int,Any}[], "failed to create $(prop.name): $(sprint(showerror, e))", Dict{Any,Any}(), nothing
            end
            push!(new_components, comp)

            # Extract unscoped symbol now, while comp is a standalone object.
            pname = guess.parameter_name
            try
                sym = getproperty(comp, pname)
                push!(comp_param_syms, i => sym)
            catch err
                @warn "Parameter extraction: could not get $(prop.name).$(pname) from standalone component — $(sprint(showerror, err))"
            end

            # Caller-supplied connector guesses, needed only when MTK
            # initialisation produces cyclic symbolic substitutions.
            for (attr, guess_val) in guess.connector_guesses
                try
                    comp_guesses[getproperty(comp, attr)] = guess_val
                catch
                end
            end

            a_sys = get(sys_dict, prop.port_a[1], nothing)
            b_sys = get(sys_dict, prop.port_b[1], nothing)
            (isnothing(a_sys) || isnothing(b_sys)) &&
                return nothing, Pair{Int,Any}[], "unknown subsystem for $(prop.name)", Dict{Any,Any}(), nothing

            a_port = getproperty(a_sys, prop.port_a[2])
            b_port = getproperty(b_sys, prop.port_b[2])
            # Connector names come from ComponentGuess.connector_names; the two
            # endpoints are wired to a_port and b_port.
            (conn_a, conn_b) = guess.connector_names
            comp_p = getproperty(comp, conn_a)
            comp_n = getproperty(comp, conn_b)

            if prop.port_a[1] == prop.port_b[1]
                # Parallel: add two new connect equations
                push!(mod_eqs, connect(a_port, comp_p))
                push!(mod_eqs, connect(b_port, comp_n))
            else
                # Series: replace the direct connection, matched by display
                # name, since ports differ in identity between the two sources.
                a_name = "$(prop.port_a[1]).$(prop.port_a[2])"
                b_name = "$(prop.port_b[1]).$(prop.port_b[2])"
                idx = findfirst(mod_eqs) do eq
                    s = _eq_to_connect_display(eq)
                    s == "connect($a_name, $b_name)" || s == "connect($b_name, $a_name)"
                end
                if !isnothing(idx)
                    mod_eqs[idx] = connect(a_port, comp_p)
                    insert!(mod_eqs, idx + 1, connect(comp_n, b_port))
                else
                    push!(mod_eqs, connect(a_port, comp_p))
                    push!(mod_eqs, connect(comp_n, b_port))
                end
            end
        end

        all_sys     = vcat(base_subsys, new_components)
        adapted_raw = ODESystem(mod_eqs, t_iv; systems = all_sys, name = :adapted_model)

        # Merge the caller's adaptation guesses with the generated connector
        # guesses; the latter take precedence.
        merged_guesses = merge(problem.adaptation_guesses, comp_guesses)
        return adapted_raw, comp_param_syms, "model built", merged_guesses, new_components
    catch e
        return nothing, Pair{Int,Any}[], "build failed: $(sprint(showerror, e))", Dict{Any,Any}(), nothing
    end
end

# Pipeline-facing wrapper returning the simplified system, which parameter
# fitting and the drift re-check need, in the original 4-tuple form.
function _build_adapted_ode_system(
    proposals::Vector{ComponentProposal},
    init_vals::Vector{Float64},
    problem::AIRMEDProblem,
)::Tuple{Union{ODESystem, Nothing}, Vector{Pair{Int,Any}}, String, Dict{Any,Any}}
    raw, syms, msg, guesses, _ = _assemble_adapted_system(proposals, init_vals, problem)
    isnothing(raw) && return nothing, syms, msg, guesses
    try
        return structural_simplify(raw), syms, msg, guesses
    catch e
        return nothing, Pair{Int,Any}[], "structural_simplify failed: $(sprint(showerror, e))", Dict{Any,Any}()
    end
end

# DriftConfig copy with a larger min_sigma, so residuals near machine
# precision do not trigger spurious detections. `method` passes through.
_with_min_sigma(c::DriftConfig, floor_val::Float64) = DriftConfig(
    method = c.method, window_size = c.window_size, min_samples = c.min_samples,
    min_sigma = max(c.min_sigma, floor_val))

# Scale the cumulative-evidence bound: `h` for CUSUM, `L` for EWMA,
# `threshold` for SimpleThreshold. A new subtype needs one line here.
_scale_evidence(m::CUSUM, scale::Real)           = CUSUM(k = m.k, h = m.h * scale)
_scale_evidence(m::EWMA, scale::Real)            = EWMA(lambda = m.lambda, L = m.L * scale)
_scale_evidence(m::SimpleThreshold, scale::Real) = SimpleThreshold(threshold = m.threshold * scale)

# DriftConfig thresholds scaled by segment over calibration length: the bounds
# are absolute, so a shorter segment would look clean through lower power.
_with_scaled_threshold(c::DriftConfig, scale::Real) = DriftConfig(
    method = _scale_evidence(c.method, scale), window_size = c.window_size,
    min_samples = c.min_samples, min_sigma = c.min_sigma)

# Shared empty-failure tuple for _fit_proposal_parameters early returns.
_fit_failure(proposals) = (proposals, false, Inf64,
                           DriftResult(false, nothing, NaN, NaN, :none, Dict{Symbol,Any}()),
                           Pair{String,Float64}[], String[])

# --- Internals: existing-parameter pre-pass (Occam's razor before structural search) ---

# Clean-fit gate: a fit is rejected when its worst channel exceeds both an
# absolute floor and a multiple of the median channel. See the documentation.
const EXISTING_FIT_WORST_CHANNEL_FLOOR     = 0.05
const EXISTING_FIT_WORST_CHANNEL_OUTLIER_K = 4.0

# Channels at the sensor noise floor are excluded from the outlier check, but
# still counted by CUSUM. See "Exclusion of noise-floor channels" in the docs.
const EXISTING_FIT_CHANNEL_SNR_MIN = 5.0

"""
    _plateau_candidates(n; min_fraction=0.15, max_fraction=0.85,
                        n_candidates=6, min_samples=8) -> Vector{UnitRange{Int}}

Trailing candidate segments of a series of length `n`, for windows that
contain a transient followed by a settled portion, which no single constant
parameter explains across both.

The boundary is not located by a test on the local derivative, which is
unreliable at realistic noise. Several tail lengths are generated instead and
the caller keeps the one that fits cleanly.

# Keywords
- `min_fraction`, `max_fraction`: shortest and longest tail, as a fraction of
  `n`. Default `0.15` and `0.85`.
- `n_candidates`: number of segments. Default `6`.
- `min_samples`: shortest admissible segment. Must exceed
  `drift_config.min_samples` by a good margin, since CUSUM spends that many
  samples on its baseline.

# Returns
The candidate segments as index ranges, longest last, or empty when `n` is too
short.
"""
function _plateau_candidates(
    n::Int;
    min_fraction::Real = 0.15,
    max_fraction::Real = 0.85,
    n_candidates::Int  = 6,
    min_samples::Int   = 8,
)
    n < 2 * min_samples && return UnitRange{Int}[]
    candidates = UnitRange{Int}[]
    seen_starts = Set{Int}()
    for frac in range(min_fraction, max_fraction; length = n_candidates)
        len = max(min_samples, round(Int, frac * n))
        len >= n && continue
        start = n - len + 1
        start in seen_starts && continue
        push!(seen_starts, start)
        push!(candidates, start:n)
    end
    return candidates
end

"""
    _fit_existing_parameters(problem, data_times, data_states;
        max_iters=300, lr=5e-2, holdout_fraction=0.25) -> Union{Nothing, NamedTuple}

Re-fit each `problem.adaptable_params` entry on its own, since fitting them
jointly risks non-identifiability. Same log-space multistart as
`_fit_proposal_parameters`, but on the unmodified model.

Every parameter is fitted against the full window and against each segment from
`_plateau_candidates`. A clean attempt wins: no residual drift, no parameter at
a bound and no grossly mismatched channel. Only if none is clean does the
lowest `fit_loss` decide, which is not comparable across segment lengths.

# Keywords
- `max_iters`, `lr`, `optimization_algorithm`: fitting settings.
- `holdout_fraction`: tail withheld from fitting.

# Returns
A `NamedTuple` `(name, param, old_value, new_value, fit_loss, drift_result,
at_bound, from_plateau)` for the best fit, or `nothing` if there is none.
`from_plateau` marks a value that explains the settled tail rather than the
whole window.
"""
function _fit_existing_parameters(
    problem::AIRMEDProblem,
    data_times::AbstractVector,
    data_states::AbstractMatrix;
    max_iters::Int = 300,
    lr::Real       = 5e-2,
    holdout_fraction::Real = 0.25,
    optimization_algorithm = OptimizationOptimJL.LBFGS(),
)
    isempty(problem.adaptable_params) && return nothing

    # CUSUM spends `min_samples` observations warming up, so a candidate needs
    # comfortably more than that to detect anything.
    candidates = _plateau_candidates(size(data_states, 2);
        min_samples = 3 * problem.drift_config.min_samples)
    if !isempty(candidates)
        @info "Existing-parameter pre-pass: also trying $(length(candidates)) candidate " *
              "trailing (plateau) segment(s) of the window (lengths " *
              "$(join([length(c) for c in candidates], ", "))/$(length(data_times)) samples), " *
              "in addition to the full window."
    end

    results = NamedTuple[]
    for ap in problem.adaptable_params
        r_full = _fit_single_existing_parameter(ap, problem, data_times, data_states;
                                                 max_iters, lr,
                                                 holdout_fraction = Float64(holdout_fraction), optimization_algorithm = optimization_algorithm)
        isnothing(r_full) || push!(results, merge(r_full, (; from_plateau = false)))

        for seg in candidates
            r_seg = _fit_single_existing_parameter(
                ap, problem, data_times[seg], data_states[:, seg];
                max_iters, lr, holdout_fraction = Float64(holdout_fraction),
                reference_n = size(data_states, 2), optimization_algorithm = optimization_algorithm)
            isnothing(r_seg) || push!(results, merge(r_seg, (; from_plateau = true)))
        end
    end
    isempty(results) && return nothing

    clean = filter(r -> !r.drift_result.drift_detected && !r.at_bound, results)
    pool  = isempty(clean) ? results : clean
    return pool[argmin([r.fit_loss for r in pool])]
end

# Fit one existing parameter as `_fit_proposal_parameters` does, without
# rebuilding the system. See "Re-anchoring for trailing plateau segments".
function _fit_single_existing_parameter(
    ap::AdaptableParam,
    problem::AIRMEDProblem,
    data_times::AbstractVector,
    data_states::AbstractMatrix;
    max_iters::Int, lr::Real, holdout_fraction::Float64, optimization_algorithm = OptimizationOptimJL.LBFGS(),
    # Sample count of the full window, which scales the drift threshold for a
    # shorter segment; see `_with_scaled_threshold`.
    reference_n::Union{Nothing, Int} = nothing,
)
    t_data = collect(Float64.(data_times))
    t0     = t_data[1]

    u0_here = if isapprox(t0, problem.tspan[1]; atol = 1e-9)
        problem.u0
    else
        row = Dict(string(s) => i for (i, s) in enumerate(problem.observable_states))
        map(problem.u0) do pr
            sym = first(pr)
            idx = get(row, string(sym), nothing)
            isnothing(idx) ? pr : (sym => data_states[idx, 1])
        end
    end
    tspan_here = (t0, problem.tspan[2])

    combined  = merge(Dict(u0_here), Dict(problem.p0))
    base_prob = try
        ODEProblem(problem.simplified_model, combined, tspan_here;
                  guesses = problem.adaptation_guesses, warn_initialize_determined = false)
    catch e
        @warn "Existing-parameter fitting ($(ap.name)): ODEProblem failed — $(sprint(showerror, e))"
        return nothing
    end

    set_p = try
        s = setp_oop(base_prob, [ap.param])
        test_prob = remake(base_prob; p = s(base_prob, [ap.default_value]))
        isapprox(test_prob.ps[ap.param], ap.default_value; rtol = 1e-8) ? s : nothing
    catch err
        @warn "Existing-parameter fitting ($(ap.name)): setp_oop unavailable — $(sprint(showerror, err))"
        nothing
    end
    isnothing(set_p) && return nothing

    make_prob(val) = remake(base_prob; p = set_p(base_prob, [val]))

    lo, hi   = log(ap.param_min), log(ap.param_max)
    x0       = clamp(log(ap.default_value), lo, hi)
    target   = data_states
    obs_syms = problem.observable_states

    n_t     = length(t_data)
    n_train = clamp(round(Int, (1 - clamp(holdout_fraction, 0.0, 0.9)) * n_t),
                    min(4, n_t), n_t)
    t_train = t_data[1:n_train]

    obs_scales = Dict{Int, Float64}()
    for (i, _) in enumerate(obs_syms)
        i > size(target, 1) && break
        obs_scales[i] = max(sqrt(mean(abs2, target[i, :])), 1e-12)
    end

    function loss_fn(x, _)
        val = exp(clamp(x[1], lo, hi))
        try
            sol = solve(make_prob(val), Rodas5P(); saveat = t_train,
                        abstol = 1e-6, reltol = 1e-6, verbose = false)
            sol.retcode ∈ (ReturnCode.Success, ReturnCode.Default) || return Inf64
            isempty(obs_syms) && return Inf64
            pred_rows = Matrix{Float64}[]
            target_rows = Matrix{Float64}[]
            for (i, s) in enumerate(obs_syms)
                i > size(target, 1) && continue
                try
                    p_row = reshape(Float64.(sol[s]), 1, :)
                    t_row = Float64.(target[i:i, 1:n_train])
                    size(p_row, 2) == size(t_row, 2) || continue
                    scale = get(obs_scales, i, 1.0)
                    push!(pred_rows, p_row ./ scale)
                    push!(target_rows, t_row ./ scale)
                catch
                end
            end
            isempty(pred_rows) && return Inf64
            return mean(abs2, reduce(vcat, pred_rows) .- reduce(vcat, target_rows))
        catch err
            @warn "Existing-parameter fitting ($(ap.name)): loss evaluation error — $(sprint(showerror, err))" maxlog=3
            return Inf64
        end
    end

    # One-dimensional log-grid multistart, inexpensive for a single parameter.
    best_x = [x0]
    best_l = loss_fn(best_x, nothing)
    for f in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
        x_try = [clamp(x0 + f * log(10.0), lo, hi)]
        l = loss_fn(x_try, nothing)
        if l < best_l
            best_l = l
            best_x = x_try
        end
    end
    isfinite(best_l) || return nothing

    optf  = OptimizationFunction(loss_fn, Optimization.AutoFiniteDiff())
    x_fit = try
        opt_prob = OptimizationProblem(optf, best_x, nothing; lb = [lo], ub = [hi])
        solve(opt_prob, optimization_algorithm; maxiters = max_iters).u
    catch e
        @warn "Existing-parameter fitting ($(ap.name)): Optimization failed — $(sprint(showerror, e))"
        best_x
    end
    loss_fn(x_fit, nothing) <= best_l || (x_fit = best_x)

    x_log = clamp(x_fit[1], lo, hi)
    x_opt = exp(x_log)
    tol   = 0.01 * max(hi - lo, 1.0)
    at_bound = (x_log <= lo + tol) || (x_log >= hi - tol)

    # Post-fit validation over the FULL series (including any held-out tail).
    fit_mse      = Inf64
    drift_result = DriftResult(false, nothing, NaN, NaN, :none, Dict{Symbol,Any}())
    try
        val_sol = solve(make_prob(x_opt), Rodas5P(); saveat = t_data,
                        abstol = 1e-8, reltol = 1e-8, verbose = false)
        if val_sol.retcode ∈ (ReturnCode.Success, ReturnCode.Default) && !isempty(obs_syms)
            mse_sum = 0.0
            n_ok    = 0
            drift_acc = zeros(length(t_data))
            n_drift   = 0
            channel_rms = Float64[]   # per-channel normalised RMS residual
            for (i, s) in enumerate(obs_syms)
                i > size(target, 1) && continue
                try
                    pred = Float64.(val_sol[s])
                    tgt  = Float64.(target[i, :])
                    length(pred) == length(tgt) || continue
                    mse_sum   += mean(abs2, pred .- tgt)
                    n_ok      += 1
                    # Normalise by the RMS of the channel, as loss_fn does, so
                    # large uninformative channels do not mask decisive ones.
                    resid_norm = abs.(pred .- tgt) ./ get(obs_scales, i, 1.0)
                    drift_acc .+= resid_norm
                    n_drift   += 1
                    # Exclude noise-floor channels from the per-channel outlier
                    # gate below; see EXISTING_FIT_CHANNEL_SNR_MIN.
                    noise_est = length(tgt) >= 2 ? std(diff(tgt)) / sqrt(2) : Inf64
                    sig       = sqrt(mean(abs2, tgt))
                    if sig > EXISTING_FIT_CHANNEL_SNR_MIN * max(noise_est, 1e-12)
                        push!(channel_rms, sqrt(mean(abs2, resid_norm)))
                    end
                catch
                end
            end
            fit_mse = n_ok > 0 ? mse_sum / n_ok : Inf64
            if n_drift > 0
                drift_cfg = _with_min_sigma(problem.drift_config, 1e-6)
                scale     = isnothing(reference_n) ? 1.0 : length(t_data) / reference_n
                scale == 1.0 || (drift_cfg = _with_scaled_threshold(drift_cfg, scale))
                drift_result = detect_drift(drift_acc ./ n_drift, drift_cfg)
            end

            # A parameter that explains the drift must fit every observable,
            # not only the average. See the gate section in the documentation.
            if !drift_result.drift_detected && length(channel_rms) >= 3
                worst = maximum(channel_rms)
                med   = median(channel_rms)
                if worst > EXISTING_FIT_WORST_CHANNEL_FLOOR &&
                        worst > EXISTING_FIT_WORST_CHANNEL_OUTLIER_K * med
                    drift_result = DriftResult(true, nothing,
                        drift_result.mean_residual, drift_result.max_residual,
                        :per_channel_gof,
                        Dict{Symbol, Any}(:worst_channel_rms   => worst,
                                          :median_channel_rms  => med,
                                          :worst_channel_index => argmax(channel_rms)))
                end
            end
        end
    catch err
        @warn "Existing-parameter fitting ($(ap.name)): validation failed — $(sprint(showerror, err))"
    end

    return (name = ap.name, param = ap.param, old_value = ap.default_value,
            new_value = x_opt, fit_loss = fit_mse, drift_result = drift_result,
            at_bound = at_bound)
end

"""
    _winnow_proposals(proposals, problem, data_times, data_states;
        max_iters, lr, holdout_fraction, optimization_algorithm,
        max_winnow_size=3, max_winnow_fits=20)
        -> (updated_proposals, success::Bool, fit_mse::Float64,
            drift_result::DriftResult, per_sensor::Vector{Pair{String,Float64}},
            bound_flags::Vector{String})

Select the simplest adequate subset of the proposals in one response, since
`_fit_proposal_parameters` alone would accept the full set even when a smaller
one explains the drift.

Proper subsets are tried in increasing size and the smallest one that fits
cleanly is accepted, i.e. no residual drift and no parameter at a bound. If
none does, the full set is fitted.

# Keywords
- `max_iters`, `lr`, `holdout_fraction`, `optimization_algorithm`: forwarded to
  the fit.
- `max_winnow_size`: largest subset tried. Default `3`.
- `max_winnow_fits`: cap on the total number of subset fits, after which the
  full set is used. Default `20`.

# Returns
As `_fit_proposal_parameters`, but for the selected subset.
"""
function _winnow_proposals(
    proposals::Vector{ComponentProposal},
    problem::AIRMEDProblem,
    data_times::AbstractVector,
    data_states::AbstractMatrix;
    max_iters::Int, lr::Real, holdout_fraction::Real,
    max_winnow_size::Int = 3,
    max_winnow_fits::Int = 20,
    optimization_algorithm = OptimizationOptimJL.LBFGS()
)
    n = length(proposals)
    if n <= 1
        return _fit_proposal_parameters(proposals, problem, data_times, data_states;
                                        max_iters, lr, holdout_fraction, optimization_algorithm = optimization_algorithm)
    end

    n_fits_done = 0
    for k in 1:min(max_winnow_size, n - 1)
        best_at_k = nothing
        for idxs in _hook_subsets(n, k)
            n_fits_done >= max_winnow_fits && break
            n_fits_done += 1
            subset = proposals[idxs]
            fit_result = _fit_proposal_parameters(subset, problem, data_times, data_states;
                                                  max_iters, lr, holdout_fraction, optimization_algorithm = optimization_algorithm)
            _, fit_ok, fit_loss, drift_result, _, bound_flags = fit_result
            clean = fit_ok && isfinite(fit_loss) && !drift_result.drift_detected && isempty(bound_flags)
            if clean && (isnothing(best_at_k) || fit_loss < best_at_k[3])
                best_at_k = fit_result
            end
        end
        if !isnothing(best_at_k)
            @info "Adaptation: winnowed $(n) proposal(s) down to $(k) (Occam's razor) — " *
                  "a smaller subset explains the drift cleanly; the rest were unjustified extras."
            return best_at_k
        end
        n_fits_done >= max_winnow_fits && break
    end

    # No proper subset cleared the drift check, so the full set is fitted,
    # which corresponds to calling `_fit_proposal_parameters` directly.
    return _fit_proposal_parameters(proposals, problem, data_times, data_states;
                                    max_iters, lr, holdout_fraction, optimization_algorithm = optimization_algorithm)
end

"""
    _fit_proposal_parameters(proposals, problem, data_times, data_states;
        max_iters, lr, holdout_fraction, optimization_algorithm)
        -> (updated_proposals, success::Bool, fit_mse::Float64,
            drift_result::DriftResult, per_sensor::Vector{Pair{String,Float64}},
            bound_flags::Vector{String})

Fit the primary parameter of each proposal, named by
`ComponentGuess.parameter_name`, by minimising the MSE between the adapted
trajectory and the measurements.

Steps: build the adapted system once; build one `ODEProblem` with a `setp_oop`
setter so each evaluation is a `remake`; run a coordinate-wise log-grid
multistart; optimise with box-constrained LBFGS in log-space, falling back to
Adam; then re-simulate the full series for the per-sensor RMSE, the drift check
and the bound flags that feed the retry loop.

# Keywords
- `max_iters`, `lr`, `optimization_algorithm`: optimiser settings.
- `holdout_fraction`: only the first `1 - holdout_fraction` of the series is
  fitted, while the checks cover all of it.

# Returns
`(proposals, success, fit_mse, drift_result, per_sensor, bound_flags)`: the
proposals with their fitted values, whether the fit converged, its MSE, the
drift check on the adapted model, the RMSE per sensor, and the parameters that
ended at a plausibility bound.
"""
function _fit_proposal_parameters(
    proposals::Vector{ComponentProposal},
    problem::AIRMEDProblem,
    data_times::AbstractVector,
    data_states::AbstractMatrix;
    max_iters::Int = 300,
    lr::Real       = 5e-2,
    holdout_fraction::Float64 = 0.25,
    optimization_algorithm = OptimizationOptimJL.LBFGS(),
)::Tuple{Vector{ComponentProposal}, Bool, Float64, DriftResult,
         Vector{Pair{String, Float64}}, Vector{String}}
    init_vals = Float64[_default_param_val(p, problem.component_guesses) for p in proposals]

    # Also returns symbolic parameter handles from the unsimplified system,
    # where the subsystem hierarchy is intact and getproperty is reliable.
    adapted_sys, comp_param_syms, build_msg, comp_guesses = _build_adapted_ode_system(proposals, init_vals, problem)
    if isnothing(adapted_sys)
        @warn "Parameter fitting: could not build adapted model — $build_msg"
        return _fit_failure(proposals)
    end

    # Built once; comp_guesses resolves the cyclic symbolic substitutions a
    # new component's connector variables can introduce.
    combined = merge(Dict(problem.u0), Dict(problem.p0))
    base_prob = try
        ODEProblem(adapted_sys, combined, problem.tspan;
                   guesses = comp_guesses, warn_initialize_determined = false)
    catch e
        @warn "Parameter fitting: ODEProblem failed — $(sprint(showerror, e))"
        return _fit_failure(proposals)
    end

    # The symbolics of the unsimplified system are the same objects as in the
    # simplified one, so remake() can match them.
    prop_syms = Any[]
    prop_idxs = Int[]
    for (i_prop, sym) in comp_param_syms
        push!(prop_syms, sym)
        push!(prop_idxs, i_prop)
    end
    if length(prop_syms) < length(proposals)
        missing = setdiff(1:length(proposals), prop_idxs)
        for i in missing
            @warn "Parameter fitting: could not locate parameter for proposal $(proposals[i].name) — skipping"
        end
    end
    isempty(prop_syms) && return _fit_failure(proposals)

    # setp_oop writes all parameter portions, including the dependent ones
    # `remake` ignores. A round-trip check falls back to a full rebuild.
    n_p    = length(prop_syms)
    set_ps = try
        s = setp_oop(base_prob, prop_syms)
        test_prob = remake(base_prob; p = s(base_prob, init_vals[prop_idxs]))
        ok = all(isapprox(test_prob.ps[prop_syms[j]], init_vals[prop_idxs][j]; rtol = 1e-8)
                 for j in 1:n_p)
        ok || @warn "Parameter fitting: setp_oop round-trip mismatch — using slow ODEProblem rebuild path."
        ok ? s : nothing
    catch err
        @warn "Parameter fitting: setp_oop unavailable ($(sprint(showerror, err))) — using slow ODEProblem rebuild path."
        nothing
    end

    make_prob(vals) = isnothing(set_ps) ?
        ODEProblem(adapted_sys,
                   merge(combined, Dict{Any, Any}(prop_syms[j] => vals[j] for j in 1:n_p)),
                   problem.tspan;
                   guesses = comp_guesses, warn_initialize_determined = false) :
        remake(base_prob; p = set_ps(base_prob, vals))

    # Log-space plausibility bounds from the matching ComponentGuess.
    lo = Float64[]
    hi = Float64[]
    for i_prop in prop_idxs
        g = _matching_guess(proposals[i_prop], problem.component_guesses)
        push!(lo, log(isnothing(g) ? 1e-12 : g.param_min))
        push!(hi, log(isnothing(g) ? 1e12  : g.param_max))
    end

    @info "Parameter fitting: optimising $(length(prop_syms)) parameter(s): $(join([string(s) for s in prop_syms], ", "))"

    x0        = clamp.(log.(init_vals[prop_idxs]), lo, hi)   # optimise in log-space
    t_data    = collect(Float64.(data_times))
    target    = data_states
    n_st      = size(target, 1)
    obs_syms  = problem.observable_states       # symbolic vars with physical sensors

    # Hold-out split: fit on the leading window, validate on the full series so
    # structural overfitting shows up in the held-out tail.
    n_t     = length(t_data)
    n_train = clamp(round(Int, (1 - clamp(holdout_fraction, 0.0, 0.9)) * n_t),
                    min(4, n_t), n_t)
    t_train = t_data[1:n_train]
    holdout_fraction > 0 && n_train < n_t &&
        @info "Parameter fitting: fitting on first $n_train of $n_t samples " *
              "($(round(100 * n_train / n_t; sigdigits=3))%); validating on full series."

    # Per-channel RMS scales: without them the MSE follows the largest
    # channels, while the smaller, identifying ones barely reach the gradient.
    obs_scales = Dict{Int, Float64}()
    for (i, _) in enumerate(obs_syms)
        i > size(target, 1) && break
        rms_i = sqrt(mean(abs2, target[i, :]))
        obs_scales[i] = max(rms_i, 1e-12)
    end
    if !isempty(obs_scales)
        scale_info = join(["obs[$i]=×$(round(1/s; sigdigits=3))"
                           for (i, s) in sort(collect(obs_scales))], "  ")
        @info "Parameter fitting: per-channel normalisation factors (÷ RMS): $scale_info"
    end

    function loss_fn(x, _)
        vals = exp.(clamp.(x, lo, hi))
        try
            # Looser tolerances than validation: the optimiser needs
            # gradients, not accurate trajectories.
            sol = solve(make_prob(vals), Rodas5P();
                        saveat  = t_train,
                        abstol  = 1e-6, reltol = 1e-6,
                        verbose = false)
            sol.retcode ∈ (ReturnCode.Success, ReturnCode.Default) || return Inf64
            # Each observable is extracted independently, so symbols removed
            # by structural_simplify do not discard the working ones.
            if !isempty(obs_syms)
                pred_rows   = Matrix{Float64}[]
                target_rows = Matrix{Float64}[]
                for (i, s) in enumerate(obs_syms)
                    i > size(target, 1) && continue
                    try
                        p_row = reshape(Float64.(sol[s]), 1, :)
                        t_row = Float64.(target[i:i, 1:n_train])
                        size(p_row, 2) == size(t_row, 2) || continue
                        # Normalise by precomputed per-channel RMS scale.
                        scale = get(obs_scales, i, 1.0)
                        push!(pred_rows,   p_row ./ scale)
                        push!(target_rows, t_row ./ scale)
                    catch
                    end
                end
                if !isempty(pred_rows)
                    return mean(abs2,
                        reduce(vcat, pred_rows) .- reduce(vcat, target_rows))
                end
            end
            # Fallback: normalised comparison of first n_st rows by index.
            arr = Array(sol)
            n   = min(size(arr, 1), n_st)
            fb_scales = [max(sqrt(mean(abs2, target[i, :])), 1e-12) for i in 1:n]
            return mean(abs2, (arr[1:n, :] .- target[1:n, 1:n_train]) ./
                              reshape(fb_scales, :, 1))
        catch err
            @warn "Parameter fitting: loss evaluation error — $(sprint(showerror, err))" maxlog=3
            return Inf64
        end
    end

    # Coordinate-wise log-grid multistart over default * 10^(-3..3), which
    # costs a few ODE solves and tolerates defaults that are far off.
    best_x = copy(x0)
    best_l = loss_fn(best_x, nothing)
    for j in 1:n_p
        for f in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
            x_try    = copy(best_x)
            x_try[j] = clamp(x0[j] + f * log(10.0), lo[j], hi[j])
            l = loss_fn(x_try, nothing)
            if l < best_l
                best_l = l
                best_x = x_try
            end
        end
    end
    if !isfinite(best_l)
        @warn "Parameter fitting: no finite loss found over the multistart grid — aborting optimisation"
        return _fit_failure(proposals)
    end
    @info "Parameter fitting: initial loss = $(round(best_l; sigdigits=4)) " *
          "(after multistart from $(join(round.(exp.(best_x); sigdigits=3), ", ")))"

    # Box-constrained LBFGS converges in tens of evaluations for 1–3 parameters
    # where Adam+FiniteDiff needed hundreds; Adam remains the fallback.
    optf  = OptimizationFunction(loss_fn, Optimization.AutoFiniteDiff())
    x_fit = try
        opt_prob = OptimizationProblem(optf, best_x, nothing; lb = lo, ub = hi)
        # Bounds on the OptimizationProblem make OptimizationOptimJL wrap this
        # in Fminbox automatically (box-constrained LBFGS).
        solve(opt_prob, optimization_algorithm; maxiters = max_iters).u
    catch e
        @warn "Parameter fitting: Optimization failed ($(sprint(showerror, e))) — falling back to Adam."
        try
            opt_prob = OptimizationProblem(optf, best_x, nothing)
            solve(opt_prob, Adam(lr); maxiters = max_iters).u
        catch e2
            @warn "Parameter fitting: optimisation failed — $(sprint(showerror, e2))"
            return _fit_failure(proposals)
        end
    end
    # Keep the multistart point if the optimiser somehow ended up worse.
    loss_fn(x_fit, nothing) <= best_l || (x_fit = best_x)

    x_log = clamp.(x_fit, lo, hi)
    x_opt = exp.(x_log)

    # A fitted value at its ComponentGuess bound means the optimiser pushed
    # outside the physical range, which indicates a wrong component type.
    bound_flags = String[]
    for j in 1:n_p
        tol = 0.01 * max(hi[j] - lo[j], 1.0)
        if x_log[j] <= lo[j] + tol
            push!(bound_flags, "$(prop_syms[j]) = $(round(x_opt[j]; sigdigits=4)) (at LOWER bound $(round(exp(lo[j]); sigdigits=3)))")
        elseif x_log[j] >= hi[j] - tol
            push!(bound_flags, "$(prop_syms[j]) = $(round(x_opt[j]; sigdigits=4)) (at UPPER bound $(round(exp(hi[j]); sigdigits=3)))")
        end
    end
    isempty(bound_flags) ||
        @warn "Parameter fitting: fitted value(s) pinned at plausibility bounds — " *
              "structure likely wrong: $(join(bound_flags, "; "))"
    updated = copy(proposals)
    for (j, i_prop) in enumerate(prop_idxs)
        prop  = proposals[i_prop]
        pname = _main_param_name(prop, problem.component_guesses)
        updated[i_prop] = ComponentProposal(
            prop.name, prop.component_type,
            prop.port_a, prop.port_b,
            Dict(pname => x_opt[j]),
            prop.rationale,
        )
    end

    # ---- Post-fit validation: re-simulate adapted model over the FULL series
    # (including any held-out tail), compute RMSE and drift ----
    fit_mse      = Inf64
    per_sensor   = Pair{String, Float64}[]
    drift_result = DriftResult(false, nothing, NaN, NaN, :none, Dict{Symbol,Any}())
    # Floor the baseline sigma in normalised units, so a near-perfect fit
    # cannot trigger spurious drift through exploding z-scores.
    drift_cfg = _with_min_sigma(problem.drift_config, 1e-6)
    try
        val_prob = make_prob(x_opt)
        val_sol  = solve(val_prob, Rodas5P();
                         saveat  = t_data,
                         abstol  = 1e-8, reltol = 1e-8,
                         verbose = false)

        if val_sol.retcode ∈ (ReturnCode.Success, ReturnCode.Default)
            log_lines = String["Parameter fitting complete — fitted values:"]
            for (j, i_prop) in enumerate(prop_idxs)
                push!(log_lines, "  $(prop_syms[j]) = $(round(x_opt[j]; sigdigits=6))")
            end
            push!(log_lines, "Validation (adapted model vs. measurement data):")

            if !isempty(obs_syms)
                mse_sum          = 0.0
                n_obs_ok         = 0
                # Scalar per-timestep residual: mean |pred − meas| across sensor channels.
                # Accumulated additively; divided by n_drift_obs before detect_drift.
                drift_acc        = zeros(length(t_data))
                n_drift_obs      = 0

                for (i, s) in enumerate(obs_syms)
                    i > size(target, 1) && begin
                        push!(log_lines, "  $(s): (no target row $i)")
                        continue
                    end
                    try
                        pred   = Float64.(val_sol[s])
                        tgt    = Float64.(target[i, :])
                        length(pred) == length(tgt) || begin
                            push!(log_lines, "  $(s): (length mismatch)")
                            continue
                        end
                        mse_i  = mean(abs2, pred .- tgt)
                        rmse_i = sqrt(mse_i)
                        mse_sum   += mse_i
                        n_obs_ok  += 1
                        push!(per_sensor, string(s) => rmse_i)
                        push!(log_lines, "  $(s): RMSE = $(round(rmse_i; sigdigits=4))")
                        # Normalise by the RMS of the channel; see
                        # _fit_single_existing_parameter.
                        drift_acc   .+= abs.(pred .- tgt) ./ get(obs_scales, i, 1.0)
                        n_drift_obs += 1
                    catch
                        push!(log_lines, "  $(s): (could not evaluate)")
                    end
                end

                fit_mse = n_obs_ok > 0 ? mse_sum / n_obs_ok : Inf64

                # Drift detection on the adapted model; remaining drift lets
                # the caller retry with another proposal.
                if n_drift_obs > 0
                    scalar_res   = drift_acc ./ n_drift_obs
                    drift_result = detect_drift(scalar_res, drift_cfg)
                    status_str   = drift_result.drift_detected ? "DRIFT DETECTED" : "no drift"
                    push!(log_lines,
                          "  Adapted-model drift check ($(drift_result.method)): $status_str" *
                          "  mean=$(round(drift_result.mean_residual; sigdigits=4))" *
                          "  max=$(round(drift_result.max_residual; sigdigits=4))")
                end
            else
                arr     = Array(val_sol)
                n       = min(size(arr, 1), n_st)
                fit_mse = mean(abs2, arr[1:n, :] .- target[1:n, :])
                # Fallback scalar residuals for drift detection
                scalar_res   = vec(mean(abs.(arr[1:n, :] .- target[1:n, :]), dims=1))
                drift_result = detect_drift(scalar_res, drift_cfg)
            end

            push!(log_lines, "  Overall MSE  = $(round(fit_mse; sigdigits=4))  " *
                             "(RMSE = $(round(sqrt(fit_mse); sigdigits=4)))")
            @info join(log_lines, "\n")
        else
            @warn "Parameter fitting: validation solve did not succeed (retcode=$(val_sol.retcode))"
        end
    catch err
        @warn "Parameter fitting: validation failed — $(sprint(showerror, err))"
    end

    return updated, true, fit_mse, drift_result, per_sensor, bound_flags
end

# --- Internals: code generation ---------------------------------------------

"""
    _generate_adaptation_code(problem, proposals) -> String

Produce the MTK source that builds the adapted system. The same subsystem in
both ports adds two `connect` lines; two different ones replace the existing
connection with two through the new component.

# Arguments
- `problem`: supplies the preamble, guesses and base equations.
- `proposals`: the components to insert.

# Returns
The source as a string.
"""
function _generate_adaptation_code(
    problem::AIRMEDProblem,
    proposals::Vector{ComponentProposal},
)::String
    # Emit only connect equations. The expanded pin equations are implied and
    # would appear twice in the generated code.
    base_eqs      = filter(_is_connect_eq, collect(equations(problem.model)))
    base_subsys   = ModelingToolkit.get_systems(problem.model)
    sys_dict      = Dict(nameof(s) => s for s in base_subsys)
    base_sys_strs = [string(nameof(s)) for s in base_subsys]

    # Per-equation tracking: include as-is, or replace with split connections.
    include_base = fill(true, length(base_eqs))
    insertions   = Dict{Int, Vector{String}}()   # replaced idx → replacement lines
    new_decls    = String[]
    extra_eqs    = String[]   # appended at the end
    notes        = String[]

    for prop in proposals
        name_str   = string(prop.name)
        a_sys_str  = string(prop.port_a[1])
        a_port_str = string(prop.port_a[2])
        b_sys_str  = string(prop.port_b[1])
        b_port_str = string(prop.port_b[2])

        guess = _matching_guess(prop, problem.component_guesses)
        pname = _main_param_name(prop, problem.component_guesses)
        pval  = get(prop.parameters, pname, _default_param_val(prop, problem.component_guesses))

        # Emit the @named declaration, using the caller's `code_template` when
        # provided and the generic form otherwise.
        if !isnothing(guess) && !isnothing(guess.code_template)
            try
                push!(new_decls, String(guess.code_template(name_str, pval)))
            catch err
                push!(new_decls, "# TODO: $(prop.component_type) for $(name_str) — code_template failed: $(sprint(showerror, err))")
            end
        elseif !isnothing(guess)
            push!(new_decls,
                  "@named $(name_str) = $(guess.component_type)(; $(pname) = $(pval))")
        else
            push!(new_decls, "# TODO: $(prop.component_type) for $(name_str) — add @named declaration (no ComponentGuess registered)")
        end

        # Connector names from the matching ComponentGuess, falling back to
        # (:p, :n).
        (conn_a, conn_b) = isnothing(guess) ? (:p, :n) : guess.connector_names

        if prop.port_a[1] == prop.port_b[1]
            # Parallel: add two connections to the new component.
            push!(extra_eqs, "connect($(name_str).$(conn_a), $(a_sys_str).$(a_port_str))")
            push!(extra_eqs, "connect($(name_str).$(conn_b), $(b_sys_str).$(b_port_str))")
        else
            # Series insertion by display-name matching; `isequal` is
            # unreliable here, as in _build_adapted_ode_system.
            a_name_full = "$(a_sys_str).$(a_port_str)"
            b_name_full = "$(b_sys_str).$(b_port_str)"
            idx = findfirst(eachindex(base_eqs)) do i
                !include_base[i] && return false
                s = _eq_to_connect_display(base_eqs[i])
                s == "connect($a_name_full, $b_name_full)" ||
                s == "connect($b_name_full, $a_name_full)"
            end
            split_a = "connect($(a_sys_str).$(a_port_str), $(name_str).$(conn_a))"
            split_b = "connect($(name_str).$(conn_b), $(b_sys_str).$(b_port_str))"
            if !isnothing(idx)
                include_base[idx] = false
                insertions[idx]   = [split_a, split_b]
            else
                push!(notes,
                    "No direct $(a_sys_str).$(a_port_str) → $(b_sys_str).$(b_port_str) " *
                    "connection found for $(name_str); appended as fallback.")
                push!(extra_eqs, split_a)
                push!(extra_eqs, split_b)
            end
        end
    end

    new_sys_names = [string(p.name) for p in proposals]
    all_sys_names = vcat(base_sys_strs, new_sys_names)

    io = IOBuffer()
    println(io, "# Auto-generated by AIRMED.propose_model_adaptation")
    println(io, "#")
    println(io, "# NOTE: this code is a PATCH, not a standalone script — it references the")
    println(io, "# base-model component objects listed below, which must already be in scope")
    println(io, "# (i.e. paste this after the @named declarations of your base-model script).")
    for line in problem.model_preamble
        println(io, line)
    end
    isempty(problem.model_preamble) || println(io)
    println(io, "@independent_variables t")
    println(io)
    println(io, "# Existing components (must be in scope — declared in your base-model script):")
    for sys in base_subsys
        println(io, "#   $(nameof(sys)) :: $(typeof(sys).name.name)")
    end
    println(io)
    println(io, "# Newly proposed components:")
    for d in new_decls
        println(io, d)
    end
    println(io)
    println(io, "adapted_eqs = [")
    for (i, eq) in enumerate(base_eqs)
        if include_base[i]
            println(io, "    $(_eq_to_connect_display(eq)),")
        else
            for s in get(insertions, i, String[])
                println(io, "    $s,")
            end
        end
    end
    for s in extra_eqs
        println(io, "    $s,")
    end
    println(io, "]")
    println(io)
    println(io, "@named adapted_model = ODESystem(adapted_eqs, t;")
    println(io, "    systems = [", join(all_sys_names, ", "), "])")
    if !isempty(notes)
        println(io)
        println(io, "# Generator notes:")
        for n in notes
            println(io, "#   - $n")
        end
    end
    return String(take!(io))
end

# Render a connect equation as "connect(a.p, b.p)", from the Connection struct
# fields where possible and from the verbose string otherwise.
function _eq_to_connect_display(eq)::String
    # --- primary: inspect the Connection value ---
    try
        conn_val = ModelingToolkit.value(eq.rhs)
        if conn_val isa ModelingToolkit.Connection
            fnames = fieldnames(typeof(conn_val))
            ports = if :systems in fnames
                getfield(conn_val, :systems)
            elseif :ports in fnames
                getfield(conn_val, :ports)
            else
                nothing
            end
            if !isnothing(ports) && !isempty(ports)
                port_strs = [replace(string(ModelingToolkit.getname(p)), "₊" => ".")
                             for p in ports]
                return "connect(" * join(port_strs, ", ") * ")"
            end
        end
    catch
    end

    # --- fallback: parse verbose "Model sys.port:" patterns ---
    s = string(eq)
    ms = collect(eachmatch(r"Model\s+([\w₊.]+)\s*:", s))
    if !isempty(ms)
        port_strs = [replace(m.captures[1], "₊" => ".") for m in ms]
        return "connect(" * join(port_strs, ", ") * ")"
    end

    # --- last resort: strip "0 ~ " prefix and normalise notation ---
    s2 = replace(replace(s, "₊" => "."), r"\(\s*t\s*\)" => "")
    m2 = match(r"^\s*0\s*~\s*(.*)", s2)
    return m2 !== nothing ? strip(m2.captures[1]) : strip(s2)
end

# --- Internals: optional model build ----------------------------------------

function _try_build_model(code::AbstractString)
    # The generated code carries its own imports from model_preamble.
    # invokelatest is required for the world age include_string creates.
    try
        mod = Module(:AIRMED_ADAPTED)
        include_string(mod, code)
        if isdefined(mod, :adapted_model)
            return Base.invokelatest(getfield, mod, :adapted_model), "model built"
        elseif isdefined(mod, :adapted_circuit)
            # Backward compatibility: older user-supplied templates may still
            # define `adapted_circuit`.  Accept either name.
            return Base.invokelatest(getfield, mod, :adapted_circuit), "model built"
        else
            return nothing, "code parsed but no `adapted_model` (or `adapted_circuit`) defined"
        end
    catch err
        return nothing, "build failed: $(sprint(showerror, err))"
    end
end
