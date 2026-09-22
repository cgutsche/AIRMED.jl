# API reference

The exported types and functions, grouped by the stage of the workflow they
belong to. The hand-written contracts, including the full keyword list of
[`propose_model_adaptation`](@ref) and the meaning of each result field, are in
[API contracts](api.md). Internal helpers are documented in
[Internals](internals.md).

```@index
Pages = ["reference.md"]
```

## Problem definition

```@docs
AIRMEDProblem
ComponentHook
ComponentGuess
AdaptableParam
HookResidualSpec
```

## Data sources

```@docs
FunctionDataSource
CSVDataSource
load_data
```

## Configuration

```@docs
DriftConfig
DriftDetectionMethod
CUSUM
EWMA
SimpleThreshold
SupervisionConfig
```

## Simulation

```@docs
simulate
simulate_ude
make_ode_problem
interpolate_to_times
model_summary
get_state_names
get_param_names
```

## Validation and drift detection

```@docs
compute_residuals
detect_drift
DriftResult
```

## Model adaptation

```@docs
propose_model_adaptation
ModelAdaptationResult
ComponentProposal
ParameterAdjustment
LLMExchange
model_adaptation_summary
explain_for_user
```

## Rebuilding the adapted twin

```@docs
build_fix_problem
build_adapted_problem
build_parameter_adjusted_problem
build_adapted_system
rebuild_problem
```

## UDE training and symbolic regression

```@docs
build_nn
train_ude
train_symbolic_ude
run_update_workflow
UpdateResult
symbolic_regression_of_nn
sparse_regression
model_update_summary
add_nn_to_ode
```

## Supervision

```@docs
AgentState
agent_supervise!
record_update!
record_adaptation!
generate_report
```

## Top-level workflow

```@docs
run_airmed
```
