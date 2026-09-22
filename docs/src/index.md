```@raw html
<p align="center">
  <img src="assets/logo.svg" alt="AIRMED.jl" width="480">
</p>
```

# AIRMED.jl

A Julia framework for self-adapting, explainable digital twins. AIRMED compares
a ModelingToolkit model against measurements, detects when the two have drifted
apart, diagnoses the cause from the sensors, and proposes a physical component
with fitted parameters that closes the gap.

!!! note "Explainability principle"
    The operating twin remains a symbolic MTK model. Neural corrections are used
    only as diagnostic instruments, to characterise a drift so that an
    explainable physical component can replace them. A network is never deployed
    as part of the running system, since a twin containing a neural correction is
    not explainable.

## Workflow

```
simulate -> compare to data -> detect drift
              | drift
     re-fit an existing parameter?  --yes--> done, without an LLM call
              | no
     locate it             (balance-law residual and SNR per hook)
     characterise it       (static-law test, switching, periodicity, sparse fit)
              |
     request a component from the caller's library
              |
     fit its parameters -> re-simulate -> re-check drift
              | still inadequate
     re-ask, listing the failed attempts (up to `max_retries`)
              | budget exhausted and still inadequate
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
  result.

## Installation

```julia
using Pkg
Pkg.develop(url = "https://github.com/cgutsche/AIRMED.jl")
```

From a clone of the repository:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite requires no LLM and no network.

## A minimal problem

```julia
using AIRMED, ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical

problem = AIRMEDProblem(;
    name  = "RC circuit",
    model = base_circuit,

    component_hooks = [
        ComponentHook(:r1_parallel, (:r1, :p), (:r1, :n), "parallel to R1"),
    ],
    component_guesses = [
        ComponentGuess(:Resistor, "Standard library resistor",
                       (name, val) -> Resistor(; R = val, name = name);
                       parameter_name = :R),
    ],
    adaptable_params = [AdaptableParam(:r1_R, r1.R, 1000.0, "R1 ageing")],

    model_preamble    = ["using ModelingToolkitStandardLibrary.Electrical"],
    data_source       = FunctionDataSource(measurements),
    tspan             = (0.0, 0.05),
    u0                = [cap.v => 0.0],
    p0                = [r1.R => 1000.0, cap.C => 1e-3, vref.k => 5.0],
    observable_states = [cap.v],
    observation_noise = [1e-3],
)

result = propose_model_adaptation(problem;
    data_times = t, data_states = x,
    api = :ollama_native, model = "granite4.2:8b-ctx8k")

fixed_problem, msg, components = build_fix_problem(problem, result)
```

## Contents

```@contents
Pages = ["workflow.md", "api.md", "design.md", "reference.md"]
Depth = 2
```

## Evaluation

Case studies, benchmarks, plotting scripts and recorded run data are maintained
in the separate AIRMED_Benchmarks repository, which depends on this package.

## License

MIT.
