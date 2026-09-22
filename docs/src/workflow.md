# AIRMED Agentic Workflow

## Overview

AIRMED runs a closed-loop iterative workflow:

```
simulate → compare → detect drift → update model → supervise → repeat
```

Each iteration is executed by `run_airmed(problem, base_ode!, nn_input_fn)`.

---

## Detailed Steps

### 1. Simulate

Run the current MTK model over the configured `tspan`.
The simplified ODE is solved with `Tsit5` (configurable via `solver=` kwarg).

### 2. Load / Generate Data

Call `load_data(problem.data_source, tspan, n_points)`.
- `FunctionDataSource`: synthetic data from a callable, e.g. for tests.
- `CSVDataSource`: reads measurement data from a CSV file.

### 3. Compute Residuals

`compute_residuals(sim_times, sim_states, data_times, data_states)`  
Linearly interpolates the simulation to measurement time points and returns
point-wise absolute residuals `|sim - meas|`.

### 4. Detect Drift

`detect_drift(scalar_residuals, drift_config)` dispatches on `drift_config.method`,
a `DriftDetectionMethod` instance that carries its own parameters:

| Method                       | Key Parameters | Notes                          |
|-------------------------------|---------------|---------------------------------|
| `CUSUM(k=.., h=..)`            | `k`, `h`      | CUSUM chart on normalised data |
| `EWMA(lambda=.., L=..)`         | `lambda`, `L` | EWMA control chart             |
| `SimpleThreshold(threshold=..)` | `threshold`   | Simple absolute threshold      |

`DriftConfig`'s remaining fields (`min_samples`, `min_sigma`, `window_size`) are
shared baseline-estimation settings, used the same way regardless of `method`.
A new algorithm requires a new `DriftDetectionMethod` subtype and one `_detect`
method in `validation.jl`; `DriftConfig` and `detect_drift` remain unchanged.

Returns a `DriftResult` with `drift_detected::Bool` and the detection index.

### 5. Supervision Decision

`agent_supervise!(state, residuals, drift, supervision_config)` returns one of:
- `:continue`: no action required
- `:update`: trigger a model update
- `:escalate`: require human review

Escalation conditions (configurable):
- Residual exceeds `human_threshold`
- Number of drift events exceeds `max_auto_updates`
- LLM agent returns `:escalate`

### 6. Model Update (UDE training), optional and diagnostic only

`run_update_workflow(...)` builds and trains a UDE:
- A Lux `Chain` network is added as an additive correction to `base_ode!`.
- Training uses `Adam` + `Optimization.AutoZygote()` + `InterpolatingAdjoint`.
- Loss: MSE between UDE trajectory and measurement data.

After training, `symbolic_regression_of_nn(...)` can sample the learned NN
to produce input/output data suitable for SINDy / DataDrivenDiffEq.

> **The trained network does not become the operating twin.** It serves as a
> diagnostic instrument for characterising a drift, after which the twin is
> rebuilt from physical components and the network is discarded. A twin
> containing a neural correction is not explainable.

This step is **optional**. `propose_model_adaptation` accepts
`update_result = nothing`, and with `include_ude_sections = false` no UDE
content reaches the LLM at all. The structural path below is self-sufficient:
it diagnoses from the sensors directly.

### 7. Structural Adaptation

When the drift is caused by a missing component rather than by a mis-tuned
parameter, `propose_model_adaptation(problem; …)` runs the following sequence:

1. **Existing-parameter pre-pass.** Each declared `AdaptableParam` is re-fitted
   independently, against the full window and against trailing plateau
   segments. If one explains the drift, the function returns without an LLM
   call. Fit quality is judged per observable channel rather than averaged,
   since an averaged score allows one badly fitted channel to be hidden by the
   remaining ones.
2. **Evidence**, computed arithmetically from the measured channels, without
   simulation and without training:
   - `_characterise_sensor_residuals`: which channels deviate from the twin;
   - `hook_snrs`: the balance-law signal-to-noise ratio at each hook, i.e. the
     location of the fault;
   - `_characterise_hooks`: the properties of the missing element at each
     active hook, i.e. its operating point, whether its law is static, and a
     sparse fit over the measured channels.
3. **Prompt assembly** from the problem definition alone. Every component type,
   port name and library label comes from the caller; the framework contributes
   no domain vocabulary.
4. **LLM call**, then parse. See [LLM backends](api.md#LLM-Backends).
5. **Winnowing.** Proposal subsets are tried in increasing size, and the
   smallest subset that fits cleanly, i.e. converged, drift cleared and no
   parameter at a bound, is selected. Winnowing can only remove proposals.
6. **Fit and re-check.** The parameters of the selected proposal are
   optimised, the adapted model is re-simulated, and drift is re-checked.
7. **Verdict.** Per-channel RMSE against the declared sensor noise. A model
   cannot fit better than its measurement, so a residual at the noise floor is
   the best attainable result.
8. **Retry (`max_retries`, default 3).** If the adapted model still drifts, if
   any channel lies above the noise floor, about 1.12 sigma for a 200-sample
   window, or if a fitted parameter is at a plausibility bound, the model is
   asked again. The re-ask consists of the same prompt plus a list of the
   attempts already made, each with its proposals, fit loss, drift verdict,
   per-channel RMSE and bound flags. It contains no simulation output and no
   optimiser trace. Once the budget is spent, the best attempt is selected,
   ranked by `fit_loss * (1 + complexity_penalty * n_components)`.
9. **Escalation (opt-in).** With `escalate_on_inadequate_structure = true`, a
   best attempt whose worst channel remains above 5 sigma, or whose fit
   produced no loss at all, is rejected rather than returned: `success = false`,
   and a separate LLM call fills `ModelAdaptationResult.escalation` with an
   account addressed to a human. Disabled by default, so that results remain
   comparable with earlier runs.

Steps 8 and 9 use different thresholds: the re-ask threshold, about 1.12 sigma,
is stricter than the escalation threshold of 5 sigma. A correct proposal
slightly above the noise floor is therefore re-asked although it would not
escalate. The additional call costs time but cannot degrade the result, since
the best attempt is still selected.

The rebuilt twin is then available through `build_fix_problem(problem, result)`.

### 8. Feedback / Validation

After an update:
- The user (or agent) can re-run the workflow to verify the updated model
  reduces drift below the configured threshold.
- `record_update!(state, result)` stores the result in `AgentState`.
- `generate_report(state)` produces a full textual supervision log.

---

## Human-in-the-Loop

Set `supervision_config.ask_human_on_uncertainty = true` to require human
confirmation before updating when residuals are large.

When escalation fires, inspect `generate_report(state)` and decide manually
whether to re-tune the UDE, revise the model structure, or accept the deviation.

---

## LLM Agent Integration

Set `supervision_config.use_agent = true` and choose an `agent_api`:
- `:anthropic`: Anthropic Messages API.
- `:openai`, `:ollama`, `:groq`, `:openrouter`: all use the OpenAI
  chat/completions format; `agent_base_url` selects the server.
- `:none`: no LLM is called. The supervisor does not escalate via the agent,
  adaptation falls back to demo-mode enumeration, and `explain_for_user`
  returns a stub.

The LLM receives the current residual statistics and the drift history and
answers with `"update"`, `"escalate"` or `"continue"`. An unrecognised or failed
response results in `:escalate`.

`agent_supervise!` evaluates `escalate_on_repeated_drift` and `human_threshold`
before consulting the agent. The agent decides only the cases that neither has
already escalated and cannot override them.

**Adaptation supports more backends than supervision.**
`propose_model_adaptation` additionally accepts `:claude_cli`, a local Claude
Code CLI in print mode, and `:ollama_native`, the native `/api/chat` endpoint of
Ollama required for reasoning models. The `agent_*` fields of the supervision
config provide the defaults, including `agent_num_ctx` and `agent_think` for
`:ollama_native`. See [LLM backends](api.md#LLM-Backends).

---

## Configuration Reference

See `src/types.jl` for all configurable fields on `DriftConfig` and `SupervisionConfig`.
