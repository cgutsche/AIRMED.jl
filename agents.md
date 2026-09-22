# AIRMED

**A Julia framework for autonomous digital twin simulation with continuous model validation, ML-based adaptation, and agent-in-the-loop supervision.**


> **Key Takeaway:**  
> This project is an adaptive digital twin simulation platform in Julia. It combines equation-based modeling (ModelingToolkit.jl; FMI planned), continuous model validation, ML-based model adaptation (UDEs; NeuralFMUs planned), and an agent-in-the-loop approach (LLM Agent) for monitoring and escalation.  
> **This AGENTS.md serves as the central specification for AI coding assistants like Copilot, Claude Code, and Cursor.**

---

### Project Overview

A modular simulation platform for digital twins that:
- Models physical systems using ModelingToolkit.jl/FMI,
- Continuously compares simulation results with measurement data,
- Detects model deviations and automatically updates the model using machine learning (UDEs, NeuralFMUs),
- Is supervised by an autonomous agent (LLM/Deep Learning) that escalates issues to a human when necessary.

---

### Task

- Create a program with the following functionality:
    - Adding an MTK model as a digital twin simulation model
    - Adding data (or a source containing data) to compare it to the model prediction
    - An automated workflow for detecting drifts (should be controallable via parameters)
    - An automated workflow to learn the missing behavior if a drift was detected (containing symbolic regression of learned behavior)
    - a Feedback system that checks if the adaptation was valid, either by human supervisor oder by agen agentic AI supervisor
- Create a test with a simple electrical system (A circuit with serial and parallel resistors) which should learn that there is a new parallel and serial resistor

- When creating an AIRMED-Problem, the developer has to add the following information:
  - The equation based model in MTK format
  - The positions within the model at which unknown behaviour may be inserted (`ComponentHook`)
  - Candidate component types (`ComponentGuess`), each with a `factory` function `(name, val) -> ODESystem`, so that AIRMED imports no domain-specific library such as `ModelingToolkitStandardLibrary` itself
  - Optionally, existing parameters eligible for direct re-fitting (`AdaptableParam`), which are tried before the structural search, so that a parameter drift is diagnosed as such rather than by adding a component
  - Parameters that may be optimised
  - The data source providing the measurement data
  - `observable_states`: the symbolic variables for which physical sensors exist, in the row order of the measurement matrix
  - `observation_noise`: the 1 sigma sensor noise per observable row, in the units of that row. It is the reference for every fit-quality verdict, retry decision and escalation; without it no noise floor is available
  - `hook_residuals` (`HookResidualSpec`): the conservation law that holds at each hook. Required for the hook-local step, which localises and characterises the missing element; the step is skipped if these or `observation_noise` are empty
  - A configuration for the drift detection
  - Whether to use an agentic supervisor or not, or when to ask a human supervisor


---

### Tech Stack

The versions below are those resolved by `Manifest.toml`. Update them together
with the manifest, not from the `[compat]` ranges.

| Component               | Package/Version (Manifest, 2026-09-21) |
|-------------------------|--------------------------------------|
| Language                | Julia 1.12.4                        |
| Equation-Based Modeling | ModelingToolkit.jl 9.84.0 (`[compat]` allows 9, 10, 11) |
| NeuralNet Integration   | ModelingToolkitNeuralNets.jl 1.7.0  |
| Neural Differential Eq. | Lux.jl 1.31.4                       |
| Test-only domain library| ModelingToolkitStandardLibrary.jl 2.21.1 |
| FMI Support             | FMI.jl, FMIFlux.jl: **planned, not yet implemented** |
| Sensitivity/Optimization| SciMLSensitivity.jl, Optimization.jl|
| Symbolic Regression     | Built-in STLSQ (`sparse_regression`) with caller-supplied basis terms |
| Data Handling           | CSV.jl, DataFrames.jl               |
| Agentic Supervision     | LLM Agent via API (Anthropic, OpenAI-compatible incl. Ollama/Groq/OpenRouter) |
| Adaptation LLM backends | The above plus `:ollama_native` (Ollama `/api/chat`, required for reasoning models) and `:claude_cli` (local Claude Code CLI in print mode) |

---

### Architecture Overview

**Layered Architecture:**
1. **Physical Layer:** Real-world system, sensors, actuators
2. **Data Acquisition Layer:** Data collection, preprocessing (e.g., via CSV, MQTT)
3. **Virtual Layer:** Equation-based model (ModelingToolkit.jl/FMI), optionally with embedded NNs (UDE/NeuralFMU)
4. **Validation & Adaptation Layer:** Simulation/measurement comparison, drift detection (CUSUM, ADWIN, residual monitoring), automatic model updates
5. **Supervision Layer:** Agent-in-the-loop (LLM/Deep Learning Agent), anomaly detection, escalation to human

---

### Project Structure

```plaintext
/
├── src/                     # Main modules: simulation, model, ML update, agent
│   ├── AIRMED.jl            # Module root; run_airmed end-to-end workflow
│   ├── types.jl             # AIRMEDProblem, ComponentHook, ComponentGuess, configs, results
│   ├── model.jl             # MTK model utilities
│   ├── simulation.jl        # Simulation logic + data sources
│   ├── validation.jl        # Drift detection, residual monitoring
│   ├── ml_update.jl         # UDE training, NN sampling, sparse symbolic regression
│   ├── model_adaptation.jl  # Component proposals (LLM/demo), fitting, retry/escalation, code generation
│   ├── twin_rebuild.jl      # build_fix_problem, build_adapted_*: the adapted twin as a runnable problem
│   └── agent_supervisor.jl  # Agent-in-the-loop logic
├── test/                    # Unit and integration tests (electrical fixture; no own Project.toml, uses [targets] below)
├── docs/                    # Documenter site
│   ├── make.jl              # Build script, deploys to gh-pages
│   ├── Project.toml         # Documentation environment
│   └── src/                 # index.md, workflow.md, api.md, design.md, reference.md
├── resources/               # Logo files
├── .github/workflows/       # CI: builds and deploys the documentation
├── Project.toml             # Julia environment; `[targets] test` adds the domain library
├── Manifest.toml            # Package versions
└── AGENTS.md                # This file
```

Case studies, benchmarks, plotting scripts, recorded run data and the resulting
figures are maintained in the separate **AIRMED_Benchmarks** repository, which
depends on this package.

> **Domain independence:** `src/` never imports a domain component library
> (`ModelingToolkitStandardLibrary` and similar). The only mention in `src/` is
> in a documentation string. All domain knowledge enters through the
> `AIRMEDProblem` definition: `ComponentGuess.factory`, `model_preamble`,
> `port_aliases`, `connector_guesses`, `adaptation_guesses`, and optional
> `extra_basis` terms for symbolic regression.
>
> Note that `ModelingToolkitStandardLibrary` currently appears in `[deps]` as
> well as `[extras]`. Only the tests and the evaluation scripts require it, so
> it may be reduced to a test dependency; in either case an import in `src/`
> remains excluded.

---

### Build & Test Commands

- **Activate environment:**  
  `julia --project=.`

- **Run tests:**  
  `julia --project=. -e 'using Pkg; Pkg.test()'`  
  (there is no `test/Project.toml`; the test deps come from `[targets]` in the
  root `Project.toml`)

- **Build the documentation:**  
  `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`  
  `julia --project=docs docs/make.jl`  
  The result is written to `docs/build/`; CI deploys it to the gh-pages branch.

- **Case studies and benchmarks:** in the AIRMED_Benchmarks repository.  
  Files under `src/` are included by the module and are not runnable on their
  own; `julia src/simulation.jl` fails on the types defined in `types.jl`.


---

### Coding Conventions

- **Modularization:** Each main component as its own Julia module.
- **Typing:** Public APIs should have explicit type annotations.
- **NeuralNet Integration:** Use Lux.jl/Flux.jl layers via ModelingToolkitNeuralNets.jl or FMIFlux.jl.
- **No global variables:** Pass state explicitly.
- **Measurement data:** load from the `data/raw/` directory of the evaluation repository, never overwrite it.

---

### Components & API Contracts

| Component         | API/Interface                                                          | Description                                  |
|-------------------|------------------------------------------------------------------------|---------------------------------------------|
| Full workflow     | `run_airmed(problem, base_ode!, nn_input_fn; adapt=true, …) -> (state, update_result, adaptation_result)` | Simulate → residuals → drift → supervise → UDE update → structural adaptation. Adaptation is gated on the UDE explaining ≥ `adapt_min_drift_explained` (default 0.25) of the drift, so the LLM only sees characterisations of corrections that actually captured the fault |
| Simulation        | `simulate(problem; saveat, solver, …)`                                 | Runs the base MTK model simulation          |
| Drift Detection   | `detect_drift(residuals, drift_config) -> DriftResult`                 | CUSUM / EWMA / threshold on residual series |
| Model Update      | `train_ude(base_ode!, nn, nn_input_fn, u0, tspan, data_times, data_states, p_base; …) -> UpdateResult` | Trains the NN correction (normalized, early-stopped, LBFGS-polished) |
| Symbolic Regression | `sparse_regression(inputs, outputs; basis_degree, extra_basis, …)`   | STLSQ recovery of the learned correction    |
| Model Adaptation  | `propose_model_adaptation(problem, update_result; …) -> ModelAdaptationResult` | Existing-parameter pre-pass (Occam's razor), then LLM/demo component proposals + parameter fitting + a retry loop (`max_retries = 3`) that re-asks with the failed attempts listed, and an opt-in escalation (`escalate_on_inadequate_structure`) that rejects a still-inadequate best attempt instead of shipping it |
| Agent Supervisor  | `agent_supervise!(state, residuals, drift, supervision_config) -> Symbol` | Returns `:continue` / `:update` / `:escalate` |

`run_airmed` returns a 3-tuple; older 2-variable destructuring keeps working.

---

### Continuous Model Validation & Drift Detection

- **Comparison:** Simulation results vs. measurement data (residuals)
- **Drift Detection:**  
  - Statistical tests: CUSUM (implemented); ADWIN (**planned**)  
  - Residual monitoring: Thresholds, EWMA control charts (implemented)  
  - `DriftConfig.min_sigma` floors the baseline σ so near-noise residuals cannot trigger spurious detections  
  - ML-based anomaly detection (optional, planned)
- **Trigger:** If error threshold is exceeded → model update

---

### ML-Based Model Update

- **UDE (NeuralFMU planned):**  
  - Unknown dynamics are learned via embedded NNs (Lux.jl)
  - Inputs are z-scored and outputs rescaled automatically from the data, so training is robust across domains with very different state magnitudes
  - Model structure remains intact, NN complements missing/incorrect dynamics
  - Best-iterate tracking, early stopping, and an LBFGS polish after Adam (via OptimizationOptimJL)
  - **Explainability principle:** the network correction is a diagnostic instrument that characterises the cause of a drift so that an explainable physical component can replace it. The operating digital twin remains the symbolic MTK model, base or adapted; trained networks are not deployed as part of the running system
- **Symbolic Regression:**  
  - The trained NN is sampled along the fitted trajectory (not a uniform grid) and `sparse_regression` (STLSQ) recovers a sparse expression; callers can supply domain basis terms via `extra_basis`
- **Parameter Estimation:**  
  - SciMLSensitivity.jl + Optimization.jl for continuous calibration
  - Loss: e.g., MSE between simulation and measurement
- **Existing-Parameter Pre-Pass:**
  - Before the structural search, `propose_model_adaptation` re-fits each declared `AdaptableParam`, i.e. each existing model parameter, individually, without an LLM call and by local optimisation only. A drift caused by a change of an existing parameter is then diagnosed as such: `ModelAdaptationResult.parameter_adjustments` is populated, `proposals` is empty and no component is added
  - Only if no declared `AdaptableParam` explains the drift, i.e. drift remains or the fit reaches a bound, does the structural search below follow
- **Structural Adaptation:**  
  - `propose_model_adaptation` converts the measured evidence into `ComponentProposal`s, either through an LLM, using forced tool use on `:anthropic` and JSON parsing with repair elsewhere, or through the demo-mode enumeration over hook subsets. It fits the parameters with multistart, box-constrained LBFGS and hold-out validation, and re-asks the model while the adapted twin still drifts, a channel lies above the declared sensor noise, about 1.12 sigma, or a fitted parameter reaches a plausibility bound. The re-ask carries the earlier attempts with their per-channel results and adds no simulation output. Adequacy is judged per channel against the sensor noise, not as a UDE-loss ratio and not averaged
  - **Escalation (opt-in, `escalate_on_inadequate_structure`):** once the retry budget is exhausted, a best attempt whose worst channel remains above 5 sigma, or that could not be fitted at all, is rejected rather than returned, and a separate LLM call writes an account into `ModelAdaptationResult.escalation`. Disabled by default, so that results remain comparable with earlier runs
  - A UDE is not required: `update_result` may be `nothing`, and `include_ude_sections = false` removes every UDE section from the prompt. The evidence then consists of the per-channel sensor residuals and the hook-local step, with `hook_snrs` providing the location and `_characterise_hooks` the properties of the missing element
  - **Winnowing:** proposal subsets are tried in increasing size, and the smallest subset that fits cleanly, i.e. converged, drift cleared and no parameter at a bound, is selected. Winnowing can only remove proposals, so it cannot repair a set that never contained the correct component
- **Rebuilding the fixed twin:**  
  - `build_fix_problem(problem, result)` returns the corrected model as a runnable `AIRMEDProblem`, dispatching over both outcome forms, i.e. a parameter fix or a structural fix. Parallel and series insertions are supported. See `docs/src/api.md`
- **Automation:**  
  - Model update is automatically triggered when drift is detected; `run_airmed(…; adapt = true)` chains UDE training and structural adaptation end-to-end

---

### Agent-in-the-Loop Supervision

- **Agent:**  
  - Monitors residuals, logs, model updates
  - Detects anomalies (e.g., repeated drift, unusual errors)
  - Issues warnings/escalations to humans if necessary
- **LLM/Deep Learning:**  
  - Can be integrated via API (e.g., OpenAI, Anthropic, local model)
  - Optional: Decision support, log interpretation, recommendations

---

### Acceptance Criteria

- All tests must pass before merging/deployment
- Simulation results must match measurement data within defined tolerances
- Model updates are only triggered for significant drift
- Agent-in-the-loop reliably detects anomalies and escalates correctly

---

### Boundaries & Guardrails

- **Measurement data is read-only** (in the evaluation repository, under `data/raw/`)
- **No secrets or production data in the repository**
- **No automatic model updates without drift detection**
- **The agent escalates to a human under uncertainty or repeated drift**
- **No domain-library imports in `src/`**; domain knowledge enters only through the `AIRMEDProblem` definition
- **The running digital twin remains explainable.** Networks and UDEs are used only to identify and characterise the cause of a drift and are not deployed as part of the operating model. Drift causes are resolved by adopting an adapted model with fitted physical components, or by manual modelling
- **FMI:** planned, not yet supported; FMI 2.0 first

---

### Iterative/Agentic Workflow

1. **Run simulation:** With the current model and parameters
2. **Compare:** Simulation vs. measurement data (calculate residuals)
3. **Drift Detection:** Use statistical/ML methods to check for model deviation
4. **Model Update:** If drift is detected, trigger a UDE or parameter update
5. **Structural Adaptation (optional, `adapt=true`):** Propose components that explain the observed deviation, fit their parameters, validate, and retry on failure
6. **Agent Supervision:** Monitors the process, detects anomalies, escalates if needed
7. **Human Review:** If escalated by the agent

All steps are chained in `run_airmed(problem, base_ode!, nn_input_fn; adapt=true)`.

> **Detailed workflow available in:**  
> `docs/src/workflow.md`

---

### Gotchas / Known Limitations

- **FMI:** not yet implemented (planned; FMI 2.0 first, no backward simulation)
- **ModelingToolkit.jl:** Recompile the model after structural changes
- **Index-1 DAEs:** models that retain algebraic unknowns after `structural_simplify`, i.e. a non-identity mass matrix, are supported. `simulate` selects a mass-matrix solver and `run_airmed` extracts the trained states by symbol. Supply numeric `adaptation_guesses` for algebraic variables, such as branch currents and node voltages, to resolve cyclic initialisation chains
- **Generated adaptation code is a patch, not a standalone script.** It references the base-model component objects from the developer's script
- **Performance:** Large models should be implemented in-place and modularly
- **Test Environment:** Always activate the Julia environment (`--project=.`)

---

### References

- [docs/src/workflow.md](docs/src/workflow.md): the workflow in detail
- [docs/src/api.md](docs/src/api.md): API specifications
- `docs/evaluation_config.md` in the AIRMED_Benchmarks repository: configuration of the runs behind the published figures
- [ModelingToolkit.jl Documentation](https://docs.sciml.ai/ModelingToolkit/stable/)
- [FMI.jl Documentation](https://github.com/ThummeTo/FMI.jl)
- [FMIFlux.jl Documentation](https://thummeto.github.io/FMIFlux.jl/dev/)
- [SciML Overview](https://docs.sciml.ai/Overview/stable/overview/)

---

> **Note:**  
> This AGENTS.md is the central specification for AI coding assistants.  
> Please update this file for any changes to architecture, workflows, or APIs.  
> **Maximum line length: 300.**  
> **Focus: What AI tools cannot infer directly from the code.**

---