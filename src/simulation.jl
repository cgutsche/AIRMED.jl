"""
Simulation logic: run the MTK model or UDE against data, load data sources.
"""

using OrdinaryDiffEq
using CSV
using DataFrames
using Statistics

"""
    simulate(problem::AIRMEDProblem; saveat=nothing, solver=nothing, kwargs...)

Simulate the base model and return the ODE solution.

# Arguments
- `problem`: supplies the simplified model, `u0`, `p0` and `tspan`.

# Keywords
- `saveat`: time points to store.
- `solver`: `nothing` picks `Tsit5`, or `Rodas5P` for an index-1 DAE.
- `abstol`, `reltol`: solver tolerances.

# Returns
The `ODESolution`.
"""
function simulate(
    problem::AIRMEDProblem;
    saveat  = nothing,
    solver  = nothing,
    abstol  = 1e-8,
    reltol  = 1e-6,
    kwargs...,
)
    ode_prob = make_ode_problem(problem)
    if isnothing(solver)
        mm     = ode_prob.f.mass_matrix
        solver = (mm isa UniformScaling || mm == I) ? Tsit5() : Rodas5P()
    end
    kw = isnothing(saveat) ? (;) : (; saveat)
    return solve(ode_prob, solver; abstol, reltol, kw..., kwargs...)
end

"""
    simulate_ude(
        ude!::Function, u0, tspan, p;
        saveat=nothing, solver=Tsit5(), kwargs...
    )

Simulate a UDE given as a plain Julia ODE function.

# Arguments
- `ude!`: `(du, u, p, t)`, the physics with its correction term.
- `u0`, `tspan`, `p`: initial state, time span and parameters.

# Keywords
- `saveat`: time points to store.
- `solver`, `abstol`, `reltol`: solver and tolerances.

# Returns
The `ODESolution`.
"""
function simulate_ude(
    ude!::Function,
    u0,
    tspan,
    p;
    saveat  = nothing,
    solver  = Tsit5(),
    abstol  = 1e-8,
    reltol  = 1e-6,
    kwargs...,
)
    prob = ODEProblem(ude!, u0, tspan, p)
    kw = isnothing(saveat) ? (;) : (; saveat)
    return solve(prob, solver; abstol, reltol, kw..., kwargs...)
end

"""
    load_data(source::FunctionDataSource, tspan, n_points=200) -> (times, states)

Generate measurement data over `tspan` by calling `source.generator` as
`(tspan, n_points) -> (times, states)`.

# Arguments
- `source`: the generator to call.
- `tspan`: time span to cover.
- `n_points`: number of samples. Default `200`.

# Returns
`(times, states)`, with states as `(n_states x n_times)`.
"""
function load_data(
    source::FunctionDataSource,
    tspan::Tuple{<:Real, <:Real},
    n_points::Int = 200,
)
    return source.generator(tspan, n_points)
end

"""
    load_data(source::CSVDataSource, tspan, n_points=nothing) -> (times, states)

Load measurement data from the CSV file of `source`.

# Arguments
- `source`: names the file and its time and state columns.
- `tspan`: rows outside this span are dropped.

# Returns
`(times, states)`, with states as `(n_states x n_times)`.
"""
function load_data(
    source::CSVDataSource,
    tspan::Tuple{<:Real, <:Real},
    ::Int = 0,
)
    df  = CSV.read(source.filepath, DataFrame)
    col = source.time_column
    mask = (df[!, col] .>= tspan[1]) .& (df[!, col] .<= tspan[2])
    df_filtered = df[mask, :]
    times  = Float64.(df_filtered[!, col])
    states = Matrix{Float64}(df_filtered[!, source.state_columns])
    return times, states'
end

"""
    interpolate_to_times(sol, target_times) -> Matrix{Float64}

Interpolate a solution to arbitrary time points.

# Arguments
- `sol`: the solution to interpolate.
- `target_times`: the time points to evaluate at.

# Returns
An `(n_states x n_times)` matrix.
"""
function interpolate_to_times(sol, target_times::AbstractVector{<:Real})
    # Native vectorized interpolation: one call instead of a splatted hcat.
    return Array(sol(target_times))
end
