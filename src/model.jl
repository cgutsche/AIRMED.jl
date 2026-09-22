"""
Utilities for working with ModelingToolkit models in AIRMED.
"""

using ModelingToolkit
using OrdinaryDiffEq

"""
    model_summary(sys::ODESystem)

Print the states, parameters and equation count of `sys`. Returns `nothing`.
"""
function model_summary(sys::ODESystem)
    println("Model: $(nameof(sys))")
    println("  States:     ", unknowns(sys))
    println("  Parameters: ", parameters(sys))
    println("  Equations:  $(length(equations(sys)))")
end

"""
    get_state_names(sys::ODESystem) -> Vector{Symbol}

The names of all state variables of `sys`.
"""
function get_state_names(sys::ODESystem)::Vector{Symbol}
    return [Symbol(nameof(s)) for s in unknowns(sys)]
end

"""
    get_param_names(sys::ODESystem) -> Vector{Symbol}

The names of all parameters of `sys`.
"""
function get_param_names(sys::ODESystem)::Vector{Symbol}
    return [Symbol(nameof(p)) for p in parameters(sys)]
end

"""
    make_ode_problem(problem::AIRMEDProblem) -> ODEProblem

Build an `ODEProblem` from the simplified model of `problem`.

# Arguments
- `problem`: supplies the model, `u0`, `p0` and the initialisation guesses.

# Keywords
- `tspan`: time span to solve over. Defaults to that of `problem`.

# Returns
The `ODEProblem`. A non-empty `problem.adaptation_guesses` is passed to MTK as
initialisation `guesses`, which an adapted model with an under-determined
initialisation DAE requires.
"""
function make_ode_problem(problem::AIRMEDProblem; tspan=problem.tspan)
    combined = merge(Dict(problem.u0), Dict(problem.p0))
    isempty(problem.adaptation_guesses) &&
        return ODEProblem(problem.simplified_model, combined, tspan)
    return ODEProblem(problem.simplified_model, combined, tspan;
                      guesses = problem.adaptation_guesses,
                      warn_initialize_determined = false)
end

"""
    make_plain_ode_problem(
        ode!::Function, u0, tspan, p; saveat=nothing
    ) -> ODEProblem

Build an `ODEProblem` from a plain Julia ODE function.

# Arguments
- `ode!`: `(du, u, p, t)`.
- `u0`, `tspan`, `p`: initial state, time span and parameters.

# Keywords
- `saveat`: time points to store.

# Returns
The `ODEProblem`.
"""
function make_plain_ode_problem(ode!::Function, u0, tspan, p; saveat=nothing)
    kwargs = isnothing(saveat) ? (;) : (; saveat)
    return ODEProblem(ode!, u0, tspan, p; kwargs...)
end

"""
    add_nn_to_ode(base_ode!::Function, nn, nn_state_ref::Ref, input_fn::Function)

Wrap an ODE function with an additive network correction.

# Arguments
- `base_ode!`: `(du, u, p_base, t)`, the known physics.
- `nn`: the correction network.
- `nn_state_ref`: holds the Lux state from `Lux.setup`.
- `input_fn`: `(u, t) -> Vector`, the network inputs.

# Returns
An ODE function `(du, u, p, t)` whose `p` is a `ComponentArray` with a `nn`
field holding the network parameters.
"""
function add_nn_to_ode(base_ode!::Function, nn, nn_state_ref::Ref, input_fn::Function)
    function ude!(du, u, p, t)
        base_ode!(du, u, p.base, t)
        x = input_fn(u, t)
        correction, _ = Lux.apply(nn, x, p.nn, nn_state_ref[])
        @. du += correction
    end
    return ude!
end
