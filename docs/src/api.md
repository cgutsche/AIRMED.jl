# AIRMED API Contracts

## Core Types

### `AIRMEDProblem`

Constructor (keyword arguments):

| Field                | Type                          | Required | Description                                                         |
|----------------------|-------------------------------|----------|---------------------------------------------------------------------|
| `name`               | `String`                      | yes      | Problem identifier                                                  |
| `model`              | `ODESystem`                   | yes      | MTK equation-based model, unsimplified                              |
| `component_hooks`    | `Vector{ComponentHook}`       | no       | Topological positions at which an unknown component may be inserted |
| `component_guesses`  | `Vector{ComponentGuess}`      | no       | Candidate component types, each with a `factory`; see below         |
| `adaptable_params`   | `Vector{AdaptableParam}`      | no       | Existing model parameters that `propose_model_adaptation` may re-fit directly, tried before the structural search; see below |
| `optimizable_params` | `Vector{Symbol}`              | no       | Parameter names eligible for continuous calibration                 |
| `data_source`        | `DataSource`                  | yes      | Measurement data source                                             |
| `drift_config`       | `DriftConfig`                 | no       | Drift detection configuration; has defaults                         |
| `supervision_config` | `SupervisionConfig`           | no       | Agent supervision configuration; has defaults                       |
| `tspan`              | `Tuple{Float64, Float64}`     | yes      | Simulation time span `(t_start, t_end)`                             |
| `u0`                 | MTK state map or `Vector`     | yes      | Initial conditions                                                  |
| `p0`                 | MTK parameter map or `Vector` | yes      | Parameter values                                                    |
| `model_preamble`     | `Vector{String}`              | no       | `using` and `import` lines written into the generated adaptation code. Must include every domain library required to instantiate the proposed components, e.g. `"using ModelingToolkitStandardLibrary.Electrical"`. AIRMED imports no domain libraries itself. |
| `observable_states`  | `Vector` of MTK symbolic vars | no       | State variables for which physical sensors exist. The rows of `data_states` correspond to these variables in order. Only these are compared against measurement data during parameter fitting of the adapted model, so internal states of newly added components, which have no sensors, are excluded. If empty, the first `n_data` rows of `Array(sol)` are used. Example: `[cap.v]`. |
| `port_aliases`       | `Dict{String,String}`         | no       | Normalises port names emitted by an LLM to those of the caller's library, e.g. `"+" => "p"`. Empty by default, since AIRMED assumes no naming convention. |
| `component_library_label` | `String`                 | no       | Label of the component library shown in prompts, e.g. `"ModelingToolkitStandardLibrary.Electrical"`. |
| `adaptation_guesses` | `Dict{Any,Any}`               | no       | Numeric initial guesses for variables of existing components whose initialisation becomes ambiguous once a new branch is added. Complements `ComponentGuess.connector_guesses`, which covers the connectors of the new component. |
| `hook_residuals`     | `Vector{HookResidualSpec}`    | no       | Per-hook balance laws enabling the hook-local measurement. Empty by default, in which case the hook-local step is skipped. |
| `observation_noise`  | `Vector{Float64}`             | no       | 1 sigma sensor noise per row of `data_states`, in the units of that row. Required together with `hook_residuals`, since without a noise floor no signal-to-noise statement is possible. Its length must equal that of `observable_states`. |

The constructor automatically calls `structural_simplify(model)` and validates
that every `HookResidualSpec` names a declared hook and that
`observation_noise` matches `observable_states` in length.

---

### `ComponentGuess`

Describes a candidate component type that could explain an observed deviation.
The `factory` encapsulates all domain-specific library knowledge; the core of
AIRMED imports no domain package.

```julia
ComponentGuess(
    component_type :: Symbol,          # e.g. :Resistor
    description    :: String,
    factory        :: Function;        # (name::Symbol, param_val::Float64) -> ODESystem
    parameter_name    :: Symbol = :R,          # the fitted parameter's name on the component
    default_parameter :: Real   = 1.0,         # starting point when the LLM names no value
    param_min         :: Real   = 1e-12,       # fitting runs in log-space: both must be > 0
    param_max         :: Real   = 1e12,
    connector_names   :: Tuple{Symbol,Symbol} = (:p, :n),   # which ports get wired
    connector_guesses :: Dict   = Dict(),      # initial values for the new connectors
)
```

**Factory contract:** receives the desired component `name` and the initial
parameter value; returns a fully constructed, named `ODESystem`.  Example:

```julia
ComponentGuess(:Resistor, "Series resistor",
    (name, val) -> Resistor(; R = val, name = name);
    parameter_name = :R, default_parameter = 500.0,
    param_min = 10.0, param_max = 1e6,
    connector_names = (:p, :n),
    connector_guesses = Dict(:v => 230.0, :i => 0.5))
```

`connector_guesses` is required when the connector variables of a new component
appear in the default expression of another component. The MTK initialisation
DAE then contains cyclic symbolic substitutions, which a few numeric values
resolve.

---

### `ComponentHook`

Marks a topological position where an unknown component might be inserted.

```julia
ComponentHook(
    name        :: Symbol,                     # e.g. :r1_parallel
    port_a      :: Tuple{Symbol, Symbol},      # (subsystem, port), maps to .p
    port_b      :: Tuple{Symbol, Symbol},      # (subsystem, port), maps to .n
    description :: String,
)
```

The same subsystem in both ports denotes a **parallel** insertion, different
subsystems a **series** insertion, in which the existing direct connection is
replaced.

---

### `HookResidualSpec`

Declares the conservation law that holds at a hook. If the law does not close,
the imbalance is the through- or across-variable of the missing element,
obtained directly from the sensors without simulation.

```julia
HookResidualSpec(
    hook           :: Symbol,                      # must name a declared ComponentHook
    kind           :: Symbol,                      # :parallel or :series
    drive,                                         # an entry of observable_states
    response_terms :: Vector{Pair{Any,Float64}},   # observable => signed coefficient
    description    :: String = "",
)
```

With `:parallel`, the response is a flow variable from a node balance, driven by
the measured potential across the hook. With `:series`, the response is a
potential from a loop balance, driven by the measured flow through it.

**Domain-independent.** Nothing in this specification is electrical; the same
structure applies to force and velocity, torque and angular velocity, mass flow
and pressure, and heat flow and temperature. AIRMED sums signed channels and
propagates variances without naming a physical quantity.

**Supplied by the caller**, as is `ComponentGuess.factory`. Deriving it
automatically would require access to the connection-set internals of MTK and
would tie the core to one generation of that library.

```julia
HookResidualSpec(:fridge_parallel, :parallel, load_fridge.v,
                 [r_grid.i => 1.0, pv_conv.i => 1.0, bat_conv.i => 1.0,
                  load_fridge.i => -1.0, load_const.i => -1.0],
                 "extra path parallel to the consumers")
```

Requires `observable_states` and `observation_noise` to be non-empty; the
hook-local step is skipped otherwise.

---

### `AIRMED.hook_snrs(problem, data_times, data_states; smooth_w = 15) -> Dict{Symbol,Float64}`

*Not exported; call it as `AIRMED.hook_snrs`. Most callers instead set
`restrict_hooks_by_snr = true` on `propose_model_adaptation`, which applies it
internally.*


Balance-law signal-to-noise at every declared hook. For each
`HookResidualSpec`: sum the response terms with their signs, propagate the
independent sensor variances in quadrature, smooth the summed residual, and
divide by the noise floor scaled to match:

```
SNR = RMS(smooth(Σ cᵢ·xᵢ, w)) / (√(Σ (cᵢ·σᵢ)²) / √w)
```

The factor `√w` is required: the numerator is smoothed, so comparing it against
the raw per-sample sigma understates the noise floor by that factor and marks
quiet hooks as active.

An SNR near 1 means that the law closes to within measurement noise, so nothing
is missing at that hook and no further analysis of these data can establish
otherwise. The function is public so that a caller can restrict the prompt to
the positions supported by the measurements.

---

### `AdaptableParam`

An existing model parameter that `propose_model_adaptation` may re-fit
directly. This is the simplest adaptation outcome, with no new component and no
new equations. A drift caused by a change of an existing component's parameter
is then diagnosed as such rather than by inserting a component next to it.

```julia
AdaptableParam(
    name          :: Symbol,   # e.g. :r_load_R
    param,                     # the symbolic parameter, e.g. r_load.R
    default_value :: Real,     # nominal value, the starting point of the fit
    description   :: String = "";
    param_min :: Real = 1e-12, # bounds; fitting runs in log-space, both positive
    param_max :: Real = 1e12,
)
```

`propose_model_adaptation` tries every declared `AdaptableParam` individually,
since varying several jointly risks non-identifiability, and does so before
building a prompt or searching `component_hooks`. The pass makes no LLM call
and consists of local optimisation only. If one parameter explains the drift,
i.e. no residual drift after fitting and no value at a bound, it is accepted
immediately: `ModelAdaptationResult.parameter_adjustments` is populated and
`proposals` remains empty. Only if none does, the structural search over
`component_hooks` and `component_guesses` follows.

---

## Model Adaptation

### `propose_model_adaptation(problem, update_result; kwargs...) -> ModelAdaptationResult`

Diagnose a drift either as a re-fitted existing parameter, which is preferred,
see `AdaptableParam` above, or as concrete MTK component proposals.

`update_result` may be `nothing`; a UDE is not required. If none is present, or
with `include_ude_sections = false`, every section describing it is omitted from
the prompt and the diagnosis rests on the sensor residuals and the hook-local
step.

**Order of operations.** The existing-parameter pre-pass runs before any
characterisation is computed. If it resolves the drift, the function returns
immediately and a characterisation computed earlier would be discarded.

Key keyword arguments:

| Argument       | Default  | Description                                               |
|----------------|----------|-----------------------------------------------------------|
| `data_times`   | `nothing`| Measurement time vector, required for parameter fitting   |
| `data_states`  | `nothing`| Measurement state matrix `(n_states × n_times)`           |
| `api`          | `:none`  | LLM backend; see *Backends* below                         |
| `api_key`      | `""`     | API key (ignored for `:none` and local servers)           |
| `base_url`     | `""`     | Endpoint override; also carries the CLI path for `:claude_cli` |
| `model`        | `""`     | Model name passed through to the backend                  |
| `build_model`  | `false`  | If `true`, attempt to build the generated `ODESystem`     |
| `fit_params`   | `true`   | Fit primary parameter of each proposal to data            |
| `fit_iters`    | `300`    | Max LBFGS iterations for parameter fitting                |
| `fit_lr`       | `5e-2`   | Learning rate (fitting runs in log-space)                 |
| `optimization_algorithm` | `OptimizationOptimJL.LBFGS()` | Optimiser used for parameter fitting  |
| `max_retries`  | `3`      | Additional LLM attempts after a failed validation; see *Retry and escalation* |
| `holdout_fraction` | `0.25` | Trailing fraction withheld from fitting, used for validation |
| `n_samples`    | `50`     | Grid samples per input when a UDE correction is characterised (UDE path only) |
| `hook_local_analysis` | `true` | `false` keeps the balance-law residual and its SNR but omits the characterisation |
| `hook_summary` | `nothing`| Pre-computed hook-local section, used verbatim. `""` omits the section entirely; `nothing` recomputes it |
| `restrict_hooks_by_snr` | `false` | Offer only hooks whose measured SNR clears `hook_snr_min` |
| `hook_snr_min` | `3.0`    | Threshold for the above                                   |
| `include_ude_sections` | `true` | Forced `false` when `update_result === nothing`      |

Prompt content. Each of these changes what the model is told and therefore what
an evaluation measures. Both default to `false`; results are comparable only
between runs that used the same settings.

| Argument       | Default  | Description                                               |
|----------------|----------|-----------------------------------------------------------|
| `include_component_equations` | `false` | List each offered component type with its constitutive equations rather than its type name alone |
| `hook_terminal_inputs` | `false` | Fit the hook residual in the terminal quantities of the element itself, i.e. its drive, the derivative and integral of that drive, and time, instead of every measured channel. If the topology contains a slack element, the channel of another component can otherwise restate conservation and outscore the true constitutive law |

LLM transport:

| Argument       | Default  | Description                                               |
|----------------|----------|-----------------------------------------------------------|
| `llm_timeout`  | `180`    | HTTP read timeout in seconds per call. Local CPU backends require considerably more on long prompts; the evaluation scripts use `600` |
| `llm_log_file` | `nothing`| Append every prompt and response pair to this file        |
| `num_ctx`      | `nothing`| `:ollama_native` only; `nothing` falls back to `AIRMED_OLLAMA_NUM_CTX` |
| `think`        | `nothing`| `:ollama_native` only; `nothing` falls back to `AIRMED_OLLAMA_THINK` |

**Three levels of hook-local evidence** are reachable through these arguments,
and they are not equivalent:

| Level | Setting | Prompt carries |
|-------|---------|----------------|
| full | `hook_local_analysis = true` | balance law + SNR + verdict + characterisation |
| localisation only | `hook_local_analysis = false` | balance law + SNR + verdict |
| none | `hook_summary = ""` | no hook section at all |

The middle level still provides the localisation: the per-hook verdict, stating
that a residual is present or that nothing is missing at that hook, restates the
SNR in words. An ablation that treats this level as a no-evidence baseline
measures only what the characterisation adds beyond the localisation.

When `api = :none` (demo mode), each `ComponentHook` is paired with the first
`ComponentGuess`; parameter values are fitted to data when `data_times` /
`data_states` are provided.

### Retry and escalation

Each attempt consists of building the prompt, calling the LLM, parsing and
winnowing the response, fitting the parameters of the proposal, re-simulating
and re-checking drift. The pipeline then judges **structural adequacy per
observable channel against the declared sensor noise**, not against a UDE loss
and not averaged across channels:

```
structural_doubt = drift_detected || any(rmse_channel > accept_factor(n, w) × σ_channel)
```

`accept_factor` is a one-sided chi-quantile correction for the finite sample
size (`_hl_accept_factor`, about 1.12 at n = 200). A retry is triggered when
`structural_doubt` holds or a fitted parameter lies at a `ComponentGuess`
plausibility bound, provided the budget `max_retries` is not exhausted. The
re-asked prompt is the same prompt plus a list of the attempts already made,
each with its proposals, fit loss, drift verdict, per-channel RMSE and bound
flags. No simulation output and no optimiser trace is added.

Once the budget is exhausted, the best attempt is returned, ranked by
`fit_loss * (1 + complexity_penalty * n_components)`.

**The retry threshold is stricter than the escalation threshold**, about 1.12
sigma against 5 sigma. A correct proposal whose fit lies slightly above the
noise floor is therefore re-asked although it would not escalate. The additional
call costs time but cannot degrade the returned answer, since the best attempt
is still selected.

| Argument | Default | Effect |
|----------|---------|--------|
| `escalate_on_inadequate_structure` | `false` | Once the budget is exhausted, reject the best proposal if its worst channel exceeds **5 times the sensor noise**, or if fitting produced no loss at all, i.e. the model did not build or no fittable parameter was found, in which case `fit_loss = Inf` and every proposal carries its unfitted default. Then `success = false`, `message` reports the rejection, and `escalation` contains an account produced by a separate LLM call. Disabled by default, since it changes `success` for callers whose retries end on a poor verdict |
| `explain_on_failure` | `false` | If the LLM returns nothing usable, skip the demo-mode enumeration, which would pair hooks with library components without evidence while still reporting `success`, and fill `escalation` with an explanation instead |

The structure-quality verdict in `message` uses the same scale: **good** at or
below 1.5 sigma on every channel, **partial** at or below 5 sigma, and **poor**
above that.

**`ModelAdaptationResult` fields:**

| Field            | Type                              | Description                                            |
|------------------|------------------------------------|--------------------------------------------------------|
| `proposals`      | `Vector{ComponentProposal}`       | Insertion recommendations; empty if the fix consisted of parameter adjustments only |
| `parameter_adjustments` | `Vector{ParameterAdjustment}` | Existing parameters re-fitted in place, without a structural change; empty if a `ComponentProposal` was used |
| `generated_code` | `String`                          | Julia/MTK source for the adapted system; a comment-only snippet for a parameter-only fix |
| `adapted_model`  | `Union{ODESystem, Nothing}`       | The built `ODESystem`, only with `build_model = true`  |
| `prompt`         | `String`                          | The prompt that was sent, or would have been sent      |
| `raw_response`   | `String`                          | The raw response, or a stub message                    |
| `api`            | `Symbol`                          | The backend that was used                              |
| `success`        | `Bool`                            | Whether at least one proposal or parameter adjustment was produced and, with `escalate_on_inadequate_structure = true`, was not rejected as inadequate |
| `message`        | `String`                          | Status summary, including the structure-quality verdict |
| `fit_loss`       | `Float64`                         | MSE of the adapted trajectory against the measurement data after parameter fitting; `NaN` if fitting was skipped or failed |
| `conversation`   | `Vector{LLMExchange}`             | All exchanges in order, including retried and discarded attempts. The `note` of the accepted attempt is prefixed accordingly. Empty if no LLM call was made, e.g. when the pre-pass resolved the drift |
| `escalation`     | `String`                          | Account of why no adaptation could be made. Empty on ordinary paths; populated only through `escalate_on_inadequate_structure` or `explain_on_failure` |

`prompt` and `raw_response` refer to the accepted attempt; `conversation`
contains the rejected ones.

**`LLMExchange` fields:** `attempt::Int`, `prompt::String`,
`raw_response::String`, `n_parsed::Int`, `note::String`.

### `model_adaptation_summary(result) -> String`

Multi-line human-readable summary of proposals and generated code.

---

## LLM Backends

Selected by the `api` keyword. `base_url` picks the actual server; `api` is
mostly a routing label.

| `api`            | Transport | Notes |
|------------------|-----------|-------|
| `:none`          | none      | Demo mode: pairs each `ComponentHook` with the first `ComponentGuess`. No network. |
| `:anthropic`     | HTTPS     | Structured output via forced tool use, so no JSON repair is needed. |
| `:openai`, `:ollama`, `:groq`, `:openrouter` | HTTPS | All speak the OpenAI chat/completions format; `base_url` selects the server. A local server needs no `api_key`. |
| `:claude_cli`    | subprocess | A local Claude Code CLI in print mode (`claude -p`). `base_url` carries the executable path. |
| `:ollama_native` | HTTPS     | The native `/api/chat` endpoint of Ollama. Required for **reasoning models**; see below. |

### When to use `:ollama_native`

The OpenAI-compatible `/v1` shim accepts `think` and ignores it. A reasoning
model then spends its completion budget on a `reasoning` field and returns empty
`content`. `:ollama_native` posts to `/api/chat` with `think = false` and an
explicit `options.num_ctx`.

On an 8B model running on CPU, a prompt of about 4k tokens exceeded a 600 s HTTP
timeout with reasoning enabled and completed in 44 s with `think = false`. At a
generation rate of about 3 tokens per second, 2000 reasoning tokens correspond
to roughly 11 minutes.

`:ollama` is kept separate and unchanged, so that results obtained with it
remain comparable.

### Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `AIRMED_LLM_MAX_TOKENS` | `2048` | Completion budget. Increase it for reasoning models, since the reasoning draws on the same budget and a truncated answer is recorded as an unparseable response rather than a modelling error. |
| `AIRMED_OLLAMA_NUM_CTX` | `8192` | `options.num_ctx` for `:ollama_native`. Ollama otherwise allocates the declared maximum context of the model, which can reserve several times the memory required by the weights. |
| `AIRMED_CLAUDE_CLI` | `claude` | Path to the CLI for `:claude_cli`. |

---

## Rebuilding the Adapted Twin

After `propose_model_adaptation` has determined what was missing, callers
require the corrected model as a runnable `AIRMEDProblem`, for example to
re-check drift on a fresh window or to compare base and adapted predictions
against the measurement. The rebuild is mechanical, since everything it needs is
available on the problem the adaptation ran against, so it is part of the
framework rather than of each caller's script.

### `build_fix_problem(problem, adapt_result; kwargs...) -> (prob | nothing, msg, comps | nothing)`

Dispatcher over the two forms a `ModelAdaptationResult` can take. If both are
present, the parameter fix takes precedence, matching the ordering of the
pipeline. `comps` is `nothing` for a parameter fix, since no component is
created.

### `build_adapted_problem(problem, proposals; kwargs...) -> (prob | nothing, msg, comps | nothing)`

The structural fix as a runnable problem. `comps` holds the created component
objects in proposal order, so that `zip(proposals, comps)` associates a
component's current or power with the position it represents.

### `build_parameter_adjusted_problem(problem, adjustments; kwargs...) -> (prob, msg)`

The parameter-only fix: identical topology, adjusted values in `p0`.
Adjustments naming a parameter absent from `p0` are skipped and reported in
`msg`.

### `build_adapted_system(problem, proposals; init_vals = nothing) -> (sys | nothing, guesses, msg, comps | nothing)`

The rebuilt `ODESystem`, **unsimplified**. `AIRMEDProblem` applies
`structural_simplify` itself, and the raw system retains the caller's subsystem
objects, which permits symbolic indexing of the solution, e.g. `sol[load.i]`.
`init_vals` defaults to the fitted parameter value of each proposal, falling
back to the matching `ComponentGuess.default_parameter`.

### `rebuild_problem(problem; kwargs...) -> AIRMEDProblem`

Copy a problem, overriding only the named fields. All remaining fields are
carried over, so replacing the model retains the solver window, the initial
conditions and the drift settings.

Shared keyword arguments for the three `build_*_problem` functions:

| Argument | Default | Description |
|----------|---------|-------------|
| `name` | `"adapted twin"` | Name for the rebuilt problem |
| `data_source` | inherited | Replay source for the rebuilt twin |
| `observable_states` | inherited | Narrow this to evaluate the twin on fewer channels than the adaptation used |

The rebuilt twin is intended for simulation, not for a further adaptation:
`component_hooks`, `adaptable_params`, `hook_residuals` and `observation_noise`
are cleared. The last two are cleared together, since the constructor rejects a
`HookResidualSpec` naming an undeclared hook.

Series insertions are handled: the existing direct connection between the two
ports is located and replaced with two connections through the new component.

---

## Simulation

### `simulate(problem; saveat, solver, abstol, reltol, kwargs...) -> ODESolution`

Run the base MTK model. Returns a SciML `ODESolution`.

### `simulate_ude(ude!, u0, tspan, p; saveat, solver, ...) -> ODESolution`

Run a plain-Julia UDE ODE function. `p` must be a `ComponentArray` with field `nn`.

### `load_data(source::DataSource, tspan, n_points) -> (times, states)`

Load or generate measurement data. `states` is `(n_states × n_times)`.

### `interpolate_to_times(sol, target_times) -> Matrix`

Interpolate an ODE solution to arbitrary time points.

---

## Validation

### `compute_residuals(sim_times, sim_states, data_times, data_states) -> (times, residuals)`

Linear interpolation of simulation to measurement times, then `|sim - meas|`.

### `detect_drift(residuals, config::DriftConfig) -> DriftResult`

Accepts `AbstractVector` (scalar) or `AbstractMatrix` (multi-state; averaged across states).

**`DriftResult` fields:**
- `drift_detected::Bool`
- `drift_index::Union{Int, Nothing}`
- `mean_residual::Float64`
- `max_residual::Float64`
- `method::Symbol`
- `details::Dict{Symbol, Any}`: algorithm-specific data, e.g. CUSUM or EWMA statistics

---

## ML Update

### `build_nn(input_dim, hidden_dim, output_dim; depth=2) -> Lux.Chain`

Construct a fully-connected tanh network.

### `train_symbolic_ude(ude_prob, sym_nn, θ, data_times, data_states; extra_params=[], solver=Tsit5(), max_iters=5000, lr=0.01, verbose=false) -> UpdateResult`

The symbolic-UDE path: trains a UDE whose unknown term is a
`@SymbolicNeuralNetwork`-embedded Lux network already present in `ude_prob`'s
equations. Uses `AutoForwardDiff` rather than Zygote. Populates the
`sym_nn` / `sym_theta` / `fitted_ode_prob` fields of `UpdateResult`; evaluate the
learned function with
`result.fitted_ode_prob.ps[result.sym_nn](x, result.fitted_ode_prob.ps[result.sym_theta])`.

### `train_ude(base_ode!, nn, nn_input_fn, u0, tspan, data_times, data_states, p_base; max_iters, lr, ...) -> UpdateResult`

Train the UDE. `p_base` is passed directly to `base_ode!` (not differentiated).
`nn_input_fn(u, t) -> Vector{Float32}` selects NN inputs from state + time.

Additional keyword arguments:
- `normalize = true`: wrap the network with input z-scoring and output rescaling derived from the data, which makes training robust across domains. The returned `UpdateResult.nn` is the wrapped network
- `patience = 100`: stop early when the best loss has not improved for this many iterations
- `loss_target = 0.0`: stop early when the loss falls below this value
- `polish = true`, `polish_iters = 100`: LBFGS polish after the Adam phase

The iterate with the lowest loss is returned, not the last one.

**`max_iters <= 0` selects the untrained control.** No optimisation runs: both
the Adam phase and the LBFGS polish are skipped, the network is returned as
`Lux.setup` initialised it, `final_loss` is the loss at initialisation rather
than `Inf`, and `n_iterations` is `0`. A small positive `max_iters` is not an
approximation of this, since the polish condition is `final_loss > loss_target`
with `loss_target = 0.0`, so any positive value triggers a full 100-iteration
LBFGS fit that dominates the result. With `normalize = true` the data-derived
scaling still applies, so an untrained correction has a plausible magnitude and
a random shape.

`n_iterations` counts Adam callback invocations only. The polish `solve` runs
without a callback, so a result reporting `n_iterations = 300` may have received
up to `polish_iters` further quasi-Newton steps.

**`UpdateResult` fields:**
- `success::Bool`
- `final_loss::Float64`
- `n_iterations::Int`
- `nn`: Lux network, including the normalisation wrappers with `normalize=true`
- `trained_nn_params`: Lux parameter struct
- `trained_nn_state`: Lux state struct
- `message::String`
- `sym_nn`, `sym_theta`, `fitted_ode_prob`: symbolic-UDE path only, otherwise `nothing`
- `trajectory_inputs`: network inputs along the fitted trajectory, of size `(n_inputs × n_times)`, or `nothing`
- `metrics::Dict{Symbol,Any}`: `:per_state_rmse_base`, `:per_state_rmse_ude`, `:drift_explained`

### `symbolic_regression_of_nn(nn, params, state, input_ranges; n_samples, max_grid_points) -> (inputs, outputs)`

Sample the trained NN over a grid (single batched `Lux.apply`; grid capped at
`max_grid_points`). The `UpdateResult` overload prefers sampling along the
fitted trajectory (`use_trajectory=true`, plus jittered copies) over the grid.

### `sparse_regression(inputs, outputs; basis_degree=3, extra_basis=[], threshold=0.05, max_sr_iters=10, scale_columns=false) -> Vector{NamedTuple}`

STLSQ (SINDy-style) sparse symbolic recovery. `extra_basis` takes `"name" => f`
pairs (`f(x::Vector) -> Real`) so domain-specific candidate terms stay in the
caller's script. Returns per-output `(expression, r2, terms)`.

This is the symbolic-regression step of the framework, and it is not limited to
network corrections: the hook-local characterisation (`_characterise_hooks`)
applies it directly to the measured balance-law residual, with
`basis_degree = 1` and sin/cos terms supplied through `extra_basis`.

| Argument | Default | Description |
|----------|---------|-------------|
| `max_sr_iters` | `10` | Number of STLSQ prune and refit iterations |
| `scale_columns` | `false` | Normalise each basis column to unit RMS before thresholding, so that pruning compares contributions rather than raw coefficients; the returned coefficients are in the original units. Required when the inputs carry different units or ranges, for example a drive of order 10, its derivative of order 0.1 and its unbounded integral, where the threshold would otherwise prune by unit. The hook-local fit sets it to `true` |

---

## Supervision

### `agent_supervise!(state, residuals, drift, config) -> Symbol`

Returns `:continue`, `:update` or `:escalate`, and mutates `state`. Call it once
per workflow iteration.

### `record_update!(state, result)`

Log a completed `UpdateResult` into `AgentState`.

### `record_adaptation!(state, result)`

Log a completed `ModelAdaptationResult` into `AgentState.adaptation_history`.

### `generate_report(state) -> String`

Full text supervision log including iteration counts, residual history, adaptation count, and log entries.

### `explain_for_user(prompt, config::SupervisionConfig; timeout = 60) -> String`

Ask the LLM configured on the `SupervisionConfig` (`agent_api` / `agent_api_key`
/ `agent_base_url` / `agent_model`) to turn a technical finding into a short
plain-language explanation for a non-expert. Returns a bracketed stub string
when `agent_api === :none`, so the result can be printed unconditionally. This
is the call behind `ModelAdaptationResult.escalation`.

**`SupervisionConfig` fields:** `use_agent` (`false`), `agent_api` (`:none`,
with the alternatives `:anthropic`, `:openai`, `:ollama`, `:groq`,
`:openrouter`), `agent_api_key`, `agent_base_url`, `agent_model`,
`escalate_on_repeated_drift` (`true`), `max_auto_updates` (`3`),
`human_threshold` (`0.5`), `ask_human_on_uncertainty` (`true`), and, for
`:ollama_native` adaptation calls only, `agent_num_ctx` and `agent_think`
(`nothing`, falling back to the environment variables above).

`agent_supervise!` evaluates `escalate_on_repeated_drift` and
`human_threshold` before consulting the agent; the agent cannot override an
escalation decided by either.

---

## Top-Level Workflow

### `run_airmed(problem, base_ode!, nn_input_fn; kwargs...) -> (AgentState, Union{UpdateResult, Nothing}, Union{ModelAdaptationResult, Nothing})`

Execute one full AIRMED iteration (simulate → compare → detect → supervise →
update → optional structural adaptation). Returns a 3-tuple; 2-variable
destructuring (`state, result = run_airmed(…)`) keeps working.

Key keyword arguments:

| Argument       | Default | Description                                   |
|----------------|---------|-----------------------------------------------|
| `n_points`     | `200`   | Data/simulation sampling resolution           |
| `verbose`      | `true`  | Print progress to stdout                      |
| `nn_input_dim` | `2`     | Input dimension of the correction network     |
| `nn_output_dim`| auto    | Output dimension (defaults to `n_states`)     |
| `nn_hidden`    | `16`    | Hidden layer width                            |
| `nn_depth`     | `2`     | Number of hidden layers                       |
| `max_iters`    | `500`   | Maximum UDE training iterations               |
| `lr`           | `1e-3`  | Learning rate for Adam optimiser              |
| `adapt`        | `false` | Run `propose_model_adaptation` after a successful UDE update |
| `adapt_api` / `adapt_api_key` / `adapt_base_url` / `adapt_model` | from `supervision_config.agent_*` | LLM backend for the adaptation step |
| `adapt_input_ranges` | auto | Ranges for the network characterisation; defaults to the extrema of the fitted trajectory ±5% |
| `adapt_min_drift_explained` | `0.25` | Skip the adaptation, with a warning, if the UDE explained less than this fraction of the base-model drift, measured by `:drift_explained`. A correction that did not capture the drift yields an unreliable characterisation. Set `0` to disable the check |
| `adapt_kwargs` | `(;)`   | `NamedTuple` forwarded to `propose_model_adaptation` (e.g. `(; max_retries=5)`) |

**Explainability principle:** the operating digital twin remains the symbolic
MTK model, base or adapted. The trained network correction serves as a
diagnostic instrument that characterises the cause of a drift so that an
explainable physical component can replace it, and it is not simulated as part
of the running system. `make_ode_problem` forwards
`problem.adaptation_guesses` as MTK initialisation `guesses`, so an adopted
adapted model whose initialisation DAE is under-determined can be simulated
directly with `simulate(problem)`.

**Index-1 DAE support:** models whose `structural_simplify` result retains
algebraic unknowns, i.e. a non-identity mass matrix, are supported throughout.
`simulate` selects `Rodas5P` when the mass matrix is not the identity, which an
explicit `solver` overrides. `run_airmed` extracts the trained dynamic states
from the solution by symbol in the order of the `problem.u0` map, rather than by
row position in `Array(sol)`, which also contains the algebraic unknowns. The
plain-Julia `u0` vector for `train_ude` is taken from the values of the `u0`
map, whose order defines the state ordering for `base_ode!`, as the order of
`p0` does for its parameters. Such models usually require numeric
`adaptation_guesses` for their algebraic variables, for example branch currents
and node voltages, to resolve cyclic symbolic substitutions during MTK
initialisation.

### `propose_model_adaptation`: structure-quality controls

| Argument             | Default | Description                                                       |
|----------------------|---------|-------------------------------------------------------------------|
| `complexity_penalty` | `0.05`  | Attempts are ranked by `fit_loss * (1 + penalty * n_components)`  |
| `holdout_fraction`   | `0.25`  | Tail fraction excluded from fitting and included in validation     |
| `max_retries`        | `3`     | Structural retry attempts, triggered by drift, a channel above the noise floor, or a parameter at a bound |
| `retry_ratio_threshold` | `10.0` | **Inert, retained for API compatibility.** It is accepted and appears in the failed-attempt text passed to the LLM, but gates nothing. The earlier test `fit_loss / ude_loss > threshold` required a UDE and gave the wrong verdict when that UDE overfitted, reporting an accurate proposal whose channels sat at their sensor sigma as incomplete. Adequacy is now judged per channel against the declared sensor noise; see *Retry and escalation* |

`ComponentGuess` gained `param_min` / `param_max` (default `1e-12`/`1e12`,
must be positive): plausibility bounds for the fitted parameter. A fitted value
pinned at a bound is reported to the LLM and triggers a structural retry.
`DriftConfig` gained `min_sigma` (default `1e-8`): floor for the baseline σ in
CUSUM/EWMA; the adapted-model drift check floors it at `1e-6 × RMS(data)`
automatically.
