<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="resources/AIRMED_logo_dark.svg">
    <img src="resources/AIRMED_logo.svg" alt="AIRMED.jl" width="520">
  </picture>
</p>

<p align="center">
  <img alt="Julia 1.12" src="https://img.shields.io/badge/Julia-1.12-9558B2">
  <img alt="Status: research prototype" src="https://img.shields.io/badge/status-research%20prototype-orange">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://cgutsche.github.io/AIRMED.jl/dev/"><img alt="Documentation" src="https://img.shields.io/badge/docs-dev-blue"></a>
</p>

---

AIRMED is a Julia framework for **self-adapting, self-healing, and explainable 
simulation models for digital twins**. It compares a ModelingToolkit model against 
measurements, detects when the two have drifted apart, diagnoses the cause from the 
sensors, and proposes a physical component with fitted parameters that closes the gap,
using an agent-in-the-loop.

The defining constraint is:

> **The operating twin remains a symbolic MTK model.** Neural corrections are
> used only as diagnostic instruments, to characterise a drift so that an
> explainable physical component can replace them. A network is never deployed
> as part of the running system, since a twin containing a neural correction is
> not explainable.

## How it works

```
simulate → compare to data → detect drift
              ↓ drift
     re-fit an existing parameter?  ──yes──→ done (no LLM call)
              ↓ no
     locate it             (balance-law residual + SNR per hook)
     characterise it       (static-law test, switching, periodicity, sparse fit)
              ↓
     ask an LLM for a component from the caller's own library
              ↓
     fit its parameters → re-simulate → re-check drift
              ↓ still inadequate
     re-ask with the failed attempts listed (up to `max_retries`)
              ↓ budget spent and still inadequate
     escalate to a human, with the measured evidence
```

Three properties distinguish this from querying a model directly:

- **The simplest explanation is tried first.** A drifted existing parameter is
  diagnosed as such by one local optimisation, without an LLM call and without
  adding a component.
- **The evidence is measured, not learned.** The hook-local characterisation is
  classical numerics on measured channels: a conservation-law residual, its
  signal-to-noise ratio against the declared sensor noise, a nearest-neighbour
  test for hidden state, a periodogram and an STLSQ sparse fit.
- **Verdicts are per channel, against sensor noise.** A model cannot fit better
  than its measurement, so a residual at the noise floor is the best attainable
  result. An averaged score would allow one badly fitted channel to be hidden
  by the remaining ones.

## Quick start

```bash
git clone <this repository>
cd AIRMED
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite requires no LLM and no network. It uses an RC circuit whose true
behaviour contains a hidden parallel and a hidden series resistor, and covers
base simulation, drift detection with all three methods, UDE training in both
the plain and the symbolic variant, and one full `run_airmed` iteration. The
structural adaptation is exercised by the case studies in the companion
evaluation repository rather than by the tests.

Case studies, benchmarks and the scripts producing the published figures live in
the separate **AIRMED_Benchmarks** repository, which depends on this one. A
worked example there is a smart-home DC microgrid with two faults: an increased
load, resolved as a parameter drift, and an unmodelled additional consumer,
resolved structurally.

## Defining a problem

All domain-specific information enters through the problem definition; `src/`
imports no component library:

```julia
using AIRMED, ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical

problem = AIRMEDProblem(;
    name  = "RC circuit",
    model = base_circuit,                       # your MTK model, unsimplified

    # Positions at which an unknown component could sit. The same subsystem
    # twice denotes a parallel, two different subsystems a series insertion.
    component_hooks = [
        ComponentHook(:r1_parallel, (:r1, :p), (:r1, :n), "parallel to R1"),
    ],

    # Candidate types. The factory keeps library knowledge in your script.
    component_guesses = [
        ComponentGuess(:Resistor, "Standard library resistor",
                       (name, val) -> Resistor(; R = val, name = name);
                       parameter_name = :R),
    ],

    # Existing parameters to re-fit before the structural search.
    adaptable_params = [AdaptableParam(:r1_R, r1.R, 1000.0, "R1 ageing")],

    model_preamble    = ["using ModelingToolkitStandardLibrary.Electrical"],
    data_source       = FunctionDataSource(measurements),
    tspan             = (0.0, 0.05),
    u0                = [cap.v => 0.0],
    p0                = [r1.R => 1000.0, cap.C => 1e-3, vref.k => 5.0],
    observable_states = [cap.v],                # the available sensors
    observation_noise = [1e-3],                 # 1 sigma per channel, the
)                                               # reference for every verdict

result = propose_model_adaptation(problem;
    data_times = t, data_states = x,
    api = :ollama_native, model = "granite4.2:8b-ctx8k")

println(result.message)
fixed_problem, msg, components = build_fix_problem(problem, result)
```

Declaring `hook_residuals`, i.e. one conservation law per hook, together with
`observation_noise` additionally enables the hook-local step, which localises
and characterises the missing element instead of inferring it from the
residuals alone.

## LLM backends

| `api` | Transport |
|-------|-----------|
| `:none` | Demo mode: enumerates the library, no network |
| `:anthropic` | Anthropic Messages API, with structured output via forced tool use |
| `:openai`, `:ollama`, `:groq`, `:openrouter` | OpenAI-compatible chat completions; `base_url` selects the server |
| `:claude_cli` | A local Claude Code CLI in print mode |
| `:ollama_native` | The native `/api/chat` endpoint of Ollama, required for reasoning models, whose `think` flag the `/v1` shim ignores |

Local models are fully supported.

## Repository layout

```
src/                 the framework
  types.jl             AIRMEDProblem, hooks, guesses, configs, results
  model.jl             MTK model utilities
  simulation.jl        simulation and data loading
  validation.jl        residuals and drift detection (CUSUM, EWMA, threshold)
  model_adaptation.jl  pre-pass, hook analysis, prompt, LLM, fit, retry, escalation
  ml_update.jl         UDE training, sparse symbolic regression (STLSQ)
  twin_rebuild.jl      the adapted twin as a runnable problem
  agent_supervisor.jl  supervision, escalation, reporting
test/                unit and integration tests (electrical fixture)
docs/                Documenter site (make.jl, src/)
resources/           logo files
```

Case studies, benchmarks, plotting scripts and recorded run data are kept in the
separate **AIRMED_Benchmarks** repository.

## Documentation

A detailed documentation can be found [here](https://cgutsche.github.io/AIRMED.jl/dev/).
