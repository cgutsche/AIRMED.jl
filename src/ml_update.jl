"""
ML-based model update: UDE training, parameter estimation, symbolic regression.
"""

using Lux
using ComponentArrays
using SciMLSensitivity
using Optimization
using OptimizationOptimisers
# `import` (not `using`): OptimizationOptimJL re-exports Optim, whose `Adam`
# would clash with the Optimisers.Adam brought in by OptimizationOptimisers.
import OptimizationOptimJL
using OrdinaryDiffEq
using Statistics
using Random
using SymbolicIndexingInterface: setp_oop

"""
    build_nn(input_dim::Int, hidden_dim::Int, output_dim::Int; depth=2) -> Lux.Chain

Construct a dense tanh network for use as a UDE correction.

# Arguments
- `input_dim`, `hidden_dim`, `output_dim`: layer widths.

# Keywords
- `depth`: number of hidden layers. Default `2`.

# Returns
The network, a `Lux.Chain` of dense tanh layers with a linear output.
"""
function build_nn(input_dim::Int, hidden_dim::Int, output_dim::Int; depth::Int = 2)
    hidden = ntuple(_ -> Dense(hidden_dim, hidden_dim, tanh), depth - 1)
    return Chain(Dense(input_dim, hidden_dim, tanh), hidden..., Dense(hidden_dim, output_dim))
end

_params_to_f32(p::AbstractVector{<:Real})  = Float32.(p)
_params_to_f32(p::AbstractVector{<:Pair}) = Float32.([last(pair) for pair in p])

"""
    _normalized_nn(nn, base_ode!, nn_input_fn, u0, tspan, data_times, data_states, p_base)
        -> Lux.Chain

Wrap `nn` with fixed input z-scoring and output rescaling, so that one set of
hyperparameters works across state magnitudes.

# Arguments
- `nn`: the network to wrap.
- `base_ode!`, `u0`, `tspan`, `p_base`: used to simulate the base model, from
  which the input statistics are taken.
- `nn_input_fn`: maps state and time to the network inputs.
- `data_times`, `data_states`: supply the output scale, the RMS of the
  finite-difference derivative.

# Keywords
- `solver`: solver for that simulation. Default `Tsit5()`.

# Returns
The wrapped network. Throws when the simulation fails or the dimensions
disagree, after which the caller uses the unwrapped network.
"""
function _normalized_nn(nn, base_ode!::Function, nn_input_fn::Function,
                        u0::AbstractVector, tspan, data_times::AbstractVector,
                        data_states::AbstractMatrix, p_base; solver = Tsit5())
    t_data = collect(Float64.(data_times))
    prob   = ODEProblem(base_ode!, Float64.(u0), Float64.(tspan), Float64.(p_base))
    sol    = solve(prob, solver; saveat = t_data, abstol = 1e-6, reltol = 1e-6,
                   verbose = false)
    SciMLBase.successful_retcode(sol) ||
        error("base-model simulation failed (retcode=$(sol.retcode))")

    X = reduce(hcat, [Float64.(nn_input_fn(sol.u[j], sol.t[j])) for j in eachindex(sol.t)])
    μ = vec(mean(X; dims = 2))
    σ = vec(std(X; dims = 2))
    # For constant inputs (sigma close to 0, e.g. a fixed source value),
    # normalise by magnitude instead, avoiding division by a near-zero value.
    σ = [s > 1e-8 ? s : max(abs(m), 1.0) for (s, m) in zip(σ, μ)]

    D  = Float64.(data_states)
    dD = diff(D; dims = 2) ./ reshape(diff(t_data), 1, :)
    out_scale = max.(vec(sqrt.(mean(abs2, dD; dims = 2))), 1e-8)

    μ32, σ32, s32 = Float32.(μ), Float32.(σ), Float32.(out_scale)
    wrapped = Chain(
        Lux.WrappedFunction(x -> (x .- μ32) ./ σ32),
        nn,
        Lux.WrappedFunction(y -> y .* s32),
    )
    # Check dimensions here rather than during training: the network output
    # must have as many entries as the measurement matrix has rows.
    ps_t, st_t = Lux.setup(Random.default_rng(), wrapped)
    y, _ = Lux.apply(wrapped, Float32.(X[:, 1]), ps_t, st_t)
    length(y) == size(D, 1) ||
        error("NN output dim ($(length(y))) ≠ data rows ($(size(D, 1)))")
    return wrapped
end

"""
    train_ude(
        base_ode!::Function,
        nn, nn_input_fn::Function,
        u0, tspan, data_times, data_states,
        p_base;
        max_iters=500, lr=1e-3, rng=default_rng(), verbose=false,
        normalize=true, patience=100, loss_target=0.0,
        polish=true, polish_iters=100,
    ) -> UpdateResult

Train a UDE by minimising the MSE between its trajectory and the measurements.
The iterate with the lowest loss is returned, not the last one.

# Arguments
- `base_ode!`: `(du, u, p_base, t)`, the known physics.
- `nn`: the correction network.
- `nn_input_fn`: `(u, t) -> Vector`, the network inputs.
- `u0`, `tspan`: initial state and time span.
- `data_times`, `data_states`: measurements, `(n_states x n_times)`.
- `p_base`: parameters passed to `base_ode!`, not differentiated.

# Keywords
- `max_iters`: Adam iterations. `<= 0` returns the untrained network with the
  loss at initialisation. Default `500`.
- `lr`: Adam learning rate. Default `1e-3`.
- `normalize`: wrap the network with data-derived scaling. Default `true`.
- `patience`: stop after this many iterations without improvement.
- `loss_target`: stop once the loss falls below it.
- `polish`, `polish_iters`: LBFGS polish after Adam, ignored on failure.
- `solver`, `rng`, `verbose`: solver, random seed and progress output.

# Returns
An `UpdateResult` holding the trained network, the best loss, the network
inputs along the fitted trajectory and the per-state RMSE metrics.
"""
function train_ude(
    base_ode!::Function,
    nn,
    nn_input_fn::Function,
    u0::AbstractVector,
    tspan::Tuple{<:Real, <:Real},
    data_times::AbstractVector,
    data_states::AbstractMatrix,
    p_base;
    solver          = Tsit5(),
    max_iters::Int  = 500,
    lr::Real        = 1e-3,
    rng             = Random.default_rng(),
    verbose::Bool   = false,
    normalize::Bool = true,
    patience::Int   = 100,
    loss_target::Real = 0.0,
    polish::Bool    = true,
    polish_iters::Int = 100,
)::UpdateResult
    p_base_f32 = _params_to_f32(p_base)

    nn_full = nn
    if normalize
        nn_full = try
            _normalized_nn(nn, base_ode!, nn_input_fn, u0, tspan,
                           data_times, data_states, p_base_f32; solver)
        catch err
            @warn "train_ude: normalisation unavailable, training on raw scales. ($(sprint(showerror, err)))"
            nn
        end
    end

    ps, st = Lux.setup(rng, nn_full)
    st_ref = Ref(st)

    function ude!(du, u, p, t)
        base_ode!(du, u, p_base_f32, t)
        x             = nn_input_fn(u, t)
        correction, _ = Lux.apply(nn_full, x, p.nn, st_ref[])
        for i in eachindex(du)
            du[i] = du[i] + correction[i]
        end
    end

    target = Float32.(data_states)
    t_data = Float32.(data_times)

    function loss_fn(p, _)
        prob = ODEProblem(ude!, Float32.(u0), Float32.(tspan), p)
        sol  = solve(prob, solver;
                     saveat   = t_data,
                     sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
                     abstol   = 1f-4, reltol = 1f-4)
        sol.retcode == ReturnCode.Success || return Inf32
        pred = Array(sol)
        return mean(abs2, pred .- target)
    end

    init_p  = ComponentArray(nn = ps)
    optf    = OptimizationFunction(loss_fn, Optimization.AutoZygote())
    opt_prob = OptimizationProblem(optf, init_p, nothing)

    losses    = Float64[]
    best_loss = Inf
    best_iter = 0
    best_u    = init_p
    callback  = (opt_state, l) -> begin
        push!(losses, l)
        lf = Float64(l)
        if lf < best_loss
            best_loss = lf
            best_iter = length(losses)
            best_u    = copy(opt_state.u)
        end
        verbose && (length(losses) % 50 == 0) && @info "  iter $(length(losses)) loss=$l"
        # Early stopping: target reached, or no improvement for `patience` iters.
        lf <= loss_target && return true
        length(losses) - best_iter >= patience && return true
        false
    end

    # Untrained control. Branching before `solve` avoids the error a
    # non-positive `maxiters` raises; the polish would itself be a full fit.
    trained = max_iters > 0

    if trained
        result = solve(opt_prob, Adam(lr); maxiters = max_iters, callback)

        final_loss = isfinite(best_loss) ? best_loss :
                     (isempty(losses) ? Inf : Float64(last(losses)))
        nn_params_ca = isfinite(best_loss) ? best_u :
                       (result.u isa ComponentArray ? result.u : init_p)
        solver_ok = result.retcode !== ReturnCode.Failure
    else
        # One forward evaluation, so the reported loss is the loss at
        # initialisation rather than `Inf`.
        final_loss   = Float64(loss_fn(init_p, nothing))
        nn_params_ca = init_p
        solver_ok    = isfinite(final_loss)
    end

    # Two-phase optimisation: LBFGS polish from the Adam optimum.
    if trained && polish && isfinite(final_loss) && final_loss > loss_target
        try
            polish_prob = OptimizationProblem(optf, nn_params_ca, nothing)
            polished    = solve(polish_prob, OptimizationOptimJL.LBFGS();
                                maxiters = polish_iters)
            l_polished  = Float64(loss_fn(polished.u, nothing))
            if isfinite(l_polished) && l_polished < final_loss
                verbose && @info "  LBFGS polish: $(round(final_loss; sigdigits=4)) -> $(round(l_polished; sigdigits=4))"
                final_loss   = l_polished
                nn_params_ca = polished.u
            end
        catch err
            verbose && @warn "train_ude: LBFGS polish failed, keeping Adam result. ($(sprint(showerror, err)))"
        end
    end

    nn_params = nn_params_ca isa ComponentArray ? nn_params_ca.nn : ps

    # Post-training diagnostics: per-state RMSE of base model vs fitted UDE,
    # and network inputs along the fitted trajectory, for on-manifold sampling.
    metrics     = Dict{Symbol, Any}()
    traj_inputs = nothing
    try
        t64       = collect(Float64.(data_times))
        base_prob = ODEProblem(base_ode!, Float64.(u0), Float64.(tspan),
                               Float64.(p_base_f32))
        base_sol  = solve(base_prob, solver; saveat = t64,
                          abstol = 1e-6, reltol = 1e-6, verbose = false)
        ude_prob  = ODEProblem(ude!, Float32.(u0), Float32.(tspan), nn_params_ca)
        ude_sol   = solve(ude_prob, solver; saveat = Float32.(t64),
                          abstol = 1f-6, reltol = 1f-6, verbose = false)
        if SciMLBase.successful_retcode(base_sol) && SciMLBase.successful_retcode(ude_sol)
            D = Float64.(data_states)
            A_base = Array(base_sol)
            A_ude  = Float64.(Array(ude_sol))
            n = min(size(D, 1), size(A_base, 1), size(A_ude, 1))
            if size(A_base, 2) == size(D, 2) && size(A_ude, 2) == size(D, 2)
                rmse_base = [sqrt(mean(abs2, A_base[i, :] .- D[i, :])) for i in 1:n]
                rmse_ude  = [sqrt(mean(abs2, A_ude[i, :]  .- D[i, :])) for i in 1:n]
                metrics[:per_state_rmse_base] = rmse_base
                metrics[:per_state_rmse_ude]  = rmse_ude
                metrics[:drift_explained] =
                    [b > 1e-12 ? 1.0 - u / b : NaN for (u, b) in zip(rmse_ude, rmse_base)]
            end
            traj_inputs = reduce(hcat,
                [Float64.(nn_input_fn(Float64.(ude_sol.u[j]), Float64(ude_sol.t[j])))
                 for j in eachindex(ude_sol.t)])
        end
    catch err
        @warn "train_ude: post-training diagnostics failed: $(sprint(showerror, err))"
    end

    return UpdateResult(
        solver_ok && isfinite(final_loss),
        final_loss,
        length(losses),
        nn_full,
        nn_params,
        st_ref[],
        "UDE training completed: final loss = $(round(final_loss; sigdigits=4))",
        nothing, nothing, nothing,
        traj_inputs,
        metrics,
    )
end

"""
    symbolic_regression_of_nn(
        nn, nn_params, nn_state, input_ranges::Vector;
        n_samples=200, max_grid_points=20_000
    ) -> (inputs, outputs)

Sample the trained network on a grid, for symbolic regression with
[`sparse_regression`](@ref).

# Arguments
- `nn`, `nn_params`, `nn_state`: the trained network.
- `input_ranges`: `(lo, hi)` per input dimension.

# Keywords
- `n_samples`: points per dimension. Default `200`.
- `max_grid_points`: cap on the total grid, which lowers the per-dimension
  resolution as needed. Default `20_000`.

# Returns
`(inputs, outputs)` of size `(n_inputs x n_points)` and
`(n_outputs x n_points)`, evaluated in one batched call.
"""
function symbolic_regression_of_nn(
    nn,
    nn_params,
    nn_state,
    input_ranges::Vector{<:Tuple{<:Real, <:Real}};
    n_samples::Int = 200,
    max_grid_points::Int = 20_000,
)
    dim   = length(input_ranges)
    n_per = min(n_samples, max(2, floor(Int, Float64(max_grid_points)^(1 / dim))))
    pts   = [range(r[1], r[2]; length = n_per) for r in input_ranges]
    grid  = collect(Iterators.product(pts...))

    inputs = Matrix{Float64}(undef, dim, length(grid))
    for (i, coords) in enumerate(grid)
        inputs[:, i] .= coords
    end

    Y, _ = Lux.apply(nn, Float32.(inputs), nn_params, nn_state)
    outputs = Float64.(Y isa AbstractMatrix ? Y : reshape(Y, :, length(grid)))
    return inputs, outputs
end

"""
    symbolic_regression_of_nn(update_result, input_ranges;
                              n_samples=200, use_trajectory=true, jitter=0.05)

Sample the network of an `UpdateResult`.

With `use_trajectory = true` and available `trajectory_inputs`, the sample
points are those visited along the fitted trajectory plus two jittered copies,
clamped to `input_ranges`. This avoids the off-manifold regions of a uniform
grid, where the network extrapolates. Otherwise the grid is used.

# Keywords
- `n_samples`: points per dimension for the grid fallback. Default `200`.
- `use_trajectory`: prefer the fitted trajectory. Default `true`.
- `jitter`: relative spread of the jittered copies. Default `0.05`.

# Returns
`(inputs, outputs)` of size `(n_inputs x n_points)` and
`(n_outputs x n_points)`.
"""
function symbolic_regression_of_nn(
    update_result::UpdateResult,
    input_ranges::Vector{<:Tuple{<:Real, <:Real}};
    n_samples::Int = 200,
    use_trajectory::Bool = true,
    jitter::Real = 0.05,
    max_grid_points::Int = 20_000,
)
    dim = length(input_ranges)

    # Trajectory-manifold sampling (Lux path only).
    ti = update_result.trajectory_inputs
    if use_trajectory && ti isa AbstractMatrix && size(ti, 1) == dim &&
            isnothing(update_result.sym_nn) && !isnothing(update_result.nn)
        spans = Float64[r[2] - r[1] for r in input_ranges]
        rng   = Random.default_rng()
        cols  = Matrix{Float64}[Float64.(ti)]
        for _ in 1:2
            J = Float64.(ti) .+ (Float64(jitter) .* spans) .* randn(rng, size(ti)...)
            for d in 1:dim
                J[d, :] .= clamp.(J[d, :], input_ranges[d][1], input_ranges[d][2])
            end
            push!(cols, J)
        end
        inputs = reduce(hcat, cols)
        Y, _ = Lux.apply(update_result.nn, Float32.(inputs),
                         update_result.trained_nn_params,
                         update_result.trained_nn_state)
        outputs = Float64.(Y isa AbstractMatrix ? Y : reshape(Y, :, size(inputs, 2)))
        return inputs, outputs
    end

    # Grid fallback, batched for the Lux path.
    if isnothing(update_result.sym_nn) && !isnothing(update_result.nn)
        return symbolic_regression_of_nn(update_result.nn,
                                         update_result.trained_nn_params,
                                         update_result.trained_nn_state,
                                         input_ranges; n_samples, max_grid_points)
    end

    # Symbolic-NN path: per-point evaluation through the fitted ODEProblem.
    n_per = min(n_samples, max(2, floor(Int, Float64(max_grid_points)^(1 / dim))))
    pts   = [range(r[1], r[2]; length = n_per) for r in input_ranges]
    grid  = collect(Iterators.product(pts...))

    x0    = [Float64(c) for c in first(grid)]
    y0    = _eval_nn_at(update_result, x0)
    n_out = length(y0)

    inputs  = Matrix{Float64}(undef, dim, length(grid))
    outputs = Matrix{Float64}(undef, n_out, length(grid))
    for (i, coords) in enumerate(grid)
        x = [Float64(c) for c in coords]
        inputs[:, i]  .= x
        outputs[:, i] .= Float64.(_eval_nn_at(update_result, x))
    end
    return inputs, outputs
end

"""
    _eval_nn_at(update_result, x) -> Vector

Evaluate the trained network at one input vector.

# Arguments
- `update_result`: the trained network, on either the symbolic or the Lux path.
- `x`: the input vector.

# Returns
The network output.
"""
function _eval_nn_at(ur::UpdateResult, x::AbstractVector)
    if !isnothing(ur.fitted_ode_prob) && !isnothing(ur.sym_nn)
        p = ur.fitted_ode_prob
        return p.ps[ur.sym_nn](x, p.ps[ur.sym_theta])
    else
        y, _ = Lux.apply(ur.nn, Float32.(x), ur.trained_nn_params, ur.trained_nn_state)
        return y
    end
end

"""
    train_symbolic_ude(
        ude_prob, sym_nn, θ, data_times, data_states;
        extra_params=[], solver=Tsit5(), max_iters=5000, lr=0.01, verbose=false
    ) -> UpdateResult

Train a UDE whose unknown term is a `@SymbolicNeuralNetwork` network, using
`AutoForwardDiff` for the gradients.

# Arguments
- `ude_prob`: `ODEProblem` whose equations already contain `sym_nn(x, θ)`.
- `sym_nn`, `θ`: the symbolic network and its parameters.
- `data_times`, `data_states`: the measurements.

# Keywords
- `extra_params`: further parameters to fit alongside `θ`.
- `solver`, `max_iters`, `lr`, `verbose`: solver and optimiser settings.

# Returns
An `UpdateResult` whose `sym_nn`, `sym_theta` and `fitted_ode_prob` fields
evaluate the learned function:
`result.fitted_ode_prob.ps[result.sym_nn](x, result.fitted_ode_prob.ps[result.sym_theta])`.
"""
function train_symbolic_ude(
    ude_prob::ODEProblem,
    sym_nn,
    θ,
    data_times::AbstractVector,
    data_states::AbstractMatrix;
    extra_params  = [],
    solver        = Tsit5(),
    max_iters::Int = 5000,
    lr::Real       = 0.01,
    verbose::Bool  = false,
)::UpdateResult
    all_tuneable = isempty(extra_params) ? θ : [collect(extra_params); collect(θ)]
    set_ps  = setp_oop(ude_prob, all_tuneable)
    ps_init = collect(Float64, ude_prob.ps[all_tuneable])
    t_data  = collect(Float64.(data_times))
    target  = Float64.(data_states)

    function loss_fn(ps_vec, _)
        p        = set_ps(ude_prob, ps_vec)
        new_prob = remake(ude_prob; p)
        sol      = solve(new_prob, solver; saveat = t_data, verbose = false)
        SciMLBase.successful_retcode(sol) || return Inf
        pred = Array(sol)
        n    = min(size(pred, 1), size(target, 1))
        return mean(abs2, pred[1:n, :] .- target[1:n, :])
    end

    losses   = Float64[]
    callback = (ps_vec, l) -> begin
        push!(losses, l)
        verbose && (length(losses) % 100 == 0) &&
            @info "  iter $(length(losses))  loss=$l"
        false
    end

    of       = OptimizationFunction(loss_fn, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(of, ps_init, nothing)
    result   = solve(opt_prob, Adam(lr); maxiters = max_iters, callback)

    final_loss  = isempty(losses) ? Inf : last(losses)
    fitted_prob = remake(ude_prob; p = set_ps(ude_prob, result.u))

    return UpdateResult(
        isfinite(final_loss),
        final_loss,
        length(losses),
        nothing, nothing, nothing,
        "Symbolic UDE training completed: final loss = $(round(final_loss; sigdigits=4))",
        sym_nn, θ, fitted_prob,
    )
end

"""
    sparse_regression(inputs, outputs;
                      basis_degree=3, extra_basis=Pair{String,Function}[],
                      threshold=0.05, max_sr_iters=10)
        -> Vector{NamedTuple}

Sequentially-thresholded least squares (the SINDy core) over a polynomial
basis, recovering a sparse expression per output channel.

# Arguments
- `inputs`: `(n_inputs x n_points)` samples, e.g. from
  [`symbolic_regression_of_nn`](@ref).
- `outputs`: `(n_outputs x n_points)` values at those samples.

# Keywords
- `basis_degree`: maximum total degree of the monomials. Default `3`.
- `extra_basis`: additional candidate terms as `"name" => f` pairs with
  `f(x) -> Real`, which keeps domain knowledge in the caller's script.
- `threshold`: coefficients below `threshold * max|c|` are dropped and the rest
  refitted. Default `0.05`.
- `max_sr_iters`: prune and refit iterations. Default `10`.
- `scale_columns`: normalise the basis columns to unit RMS first, so pruning
  compares contributions rather than units. Default `false`.

# Returns
One entry per output channel with `expression`, `r2` and `terms`.
"""
function sparse_regression(
    inputs::AbstractMatrix,
    outputs::AbstractMatrix;
    basis_degree::Int = 3,
    extra_basis::Vector{<:Pair{<:AbstractString, <:Function}} = Pair{String, Function}[],
    threshold::Real = 0.05,
    max_sr_iters::Int = 10,
    # `_stlsq` prunes on raw magnitude, so columns of differing units prune by
    # unit. `scale_columns` normalises them and returns original units.
    scale_columns::Bool = false,
)
    names, Φ = _build_basis(Float64.(inputs), basis_degree, extra_basis)
    σ = scale_columns ?
        [max(sqrt(mean(abs2, @view Φ[:, j])), 1e-12) for j in axes(Φ, 2)] :
        ones(size(Φ, 2))
    Φw = scale_columns ? Φ ./ σ' : Φ
    results  = NamedTuple{(:expression, :r2, :terms),
                          Tuple{String, Float64, Vector{Pair{String, Float64}}}}[]

    for d in axes(outputs, 1)
        y      = Float64.(outputs[d, :])
        coeffs = _stlsq(Φw, y; threshold = Float64(threshold), iters = max_sr_iters) ./ σ

        y_pred = Φ * coeffs
        ss_res = sum(abs2, y .- y_pred)
        ss_tot = sum(abs2, y .- mean(y))
        r2     = 1.0 - ss_res / max(ss_tot, 1e-15)

        active = [names[j] => coeffs[j] for j in eachindex(coeffs) if coeffs[j] != 0.0]
        expr   = isempty(active) ? "0" :
                 join([n == "1" ? string(round(c; sigdigits = 4)) :
                                  "$(round(c; sigdigits = 4))·$n"
                       for (n, c) in active], " + ")
        push!(results, (expression = expr, r2 = r2, terms = active))
    end
    return results
end

# Monomial basis up to total degree `deg` plus caller-supplied extra terms.
# Returns (term_names, Φ) with Φ of size (n_points × n_terms).
function _build_basis(inputs::Matrix{Float64}, deg::Int,
                      extra_basis::Vector{<:Pair{<:AbstractString, <:Function}})
    dim, n_pts = size(inputs)
    exps = Vector{Int}[]
    for e in Iterators.product(ntuple(_ -> 0:deg, dim)...)
        sum(e) <= deg && push!(exps, collect(e))
    end
    sort!(exps; by = e -> (sum(e), e))

    names = String[]
    cols  = Vector{Float64}[]
    for e in exps
        if all(iszero, e)
            push!(names, "1")
            push!(cols, ones(n_pts))
        else
            parts = String[]
            col   = ones(n_pts)
            for d in 1:dim
                e[d] == 0 && continue
                push!(parts, e[d] == 1 ? "x$d" : "x$d^$(e[d])")
                col .*= inputs[d, :] .^ e[d]
            end
            push!(names, join(parts, "*"))
            push!(cols, col)
        end
    end
    for (name, f) in extra_basis
        try
            push!(cols, [Float64(f(inputs[:, i])) for i in 1:n_pts])
            push!(names, String(name))
        catch err
            @warn "sparse_regression: extra basis term $(repr(name)) failed and was skipped. ($(sprint(showerror, err)))"
        end
    end
    return names, reduce(hcat, cols)
end

# Sequentially-thresholded least squares: fit, set coefficients small relative
# to the largest to zero, refit on the active support, repeat to a fixpoint.
function _stlsq(Φ::Matrix{Float64}, y::Vector{Float64};
                threshold::Float64 = 0.05, iters::Int = 10)
    coeffs = Φ \ y
    for _ in 1:iters
        cmax = maximum(abs, coeffs; init = 0.0)
        cmax == 0.0 && break
        small = abs.(coeffs) .< threshold * cmax
        any(small) || break
        coeffs[small] .= 0.0
        active = findall(!iszero, coeffs)
        isempty(active) && break
        coeffs[active] = Φ[:, active] \ y
    end
    return coeffs
end

"""
    model_update_summary(result::UpdateResult; input_ranges=nothing, n_samples=50) -> String

Summarise a trained update as text: status, loss, architecture and parameter
count.

# Arguments
- `result`: the update to summarise.

# Keywords
- `input_ranges`: one `(lo, hi)` per input dimension. Given, the summary also
  reports the range of the learned correction.
- `n_samples`: grid points used for that range. Default `50`.

# Returns
The summary as a multi-line string.
"""
function model_update_summary(
    result::UpdateResult;
    input_ranges::Union{Nothing, Vector{<:Tuple{<:Real, <:Real}}} = nothing,
    n_samples::Int = 50,
)::String
    io = IOBuffer()

    status = result.success ? "success" : "failed"
    println(io, "=== AIRMED Updated Model ===")
    println(io, "Status:     $status")
    println(io, "Loss:       $(round(result.final_loss; sigdigits=4)) " *
                "($(result.n_iterations) iterations)")
    println(io, "")
    println(io, "Neural network correction (UDE):")
    if !isnothing(result.nn)
        n_params = length(ComponentArray(result.trained_nn_params))
        println(io, "  Trainable parameters: $n_params")
        println(io, "  Architecture:")
        for (i, layer) in enumerate(result.nn.layers)
            println(io, "    [$i] $layer")
        end
    elseif !isnothing(result.sym_nn)
        println(io, "  (symbolic NN via @SymbolicNeuralNetwork)")
    end

    # Per-state fit diagnostics: the fraction of the base-model drift
    # explained by the correction.
    m = result.metrics
    if haskey(m, :per_state_rmse_base) && haskey(m, :per_state_rmse_ude)
        println(io, "")
        println(io, "Fit vs measurement data (per state):")
        rb = m[:per_state_rmse_base]
        ru = m[:per_state_rmse_ude]
        ex = get(m, :drift_explained, fill(NaN, length(rb)))
        for i in eachindex(rb)
            ex_str = isnan(ex[i]) ? "n/a" : "$(round(100 * ex[i]; sigdigits=3))%"
            println(io, "  state[$i]: RMSE base=$(round(rb[i]; sigdigits=4))  " *
                        "UDE=$(round(ru[i]; sigdigits=4))  " *
                        "drift explained=$ex_str")
        end
    end

    if !isnothing(input_ranges)
        println(io, "")
        println(io, "NN correction on $(n_samples)-point sample grid:")
        for (d, r) in enumerate(input_ranges)
            println(io, "  input[$d] ∈ [$(r[1]), $(r[2])]")
        end
        inputs, outputs = symbolic_regression_of_nn(result, input_ranges; n_samples)
        for d in axes(outputs, 1)
            v = outputs[d, :]
            println(io, "  output[$d]: " *
                        "min=$(round(minimum(v); sigdigits=4))  " *
                        "max=$(round(maximum(v); sigdigits=4))  " *
                        "mean=$(round(mean(v); sigdigits=4))")
        end

        println(io, "")
        println(io, "Symbolic regression (linear basis [1, x₁, x₂, …]):")
        _print_linear_fit!(io, inputs, outputs)

        println(io, "")
        println(io, "Sparse symbolic regression (STLSQ, polynomial basis):")
        try
            for (d, fit) in enumerate(sparse_regression(inputs, outputs))
                println(io, "  output[$d] ≈ $(fit.expression)   R²=$(round(fit.r2; sigdigits=3))")
            end
        catch err
            println(io, "  (sparse regression failed: $(sprint(showerror, err)))")
        end
    end

    return String(take!(io))
end

function _print_linear_fit!(io::IO, inputs::Matrix{Float64}, outputs::Matrix{Float64})
    n_pts = size(inputs, 2)
    n_in  = size(inputs, 1)
    A = hcat(ones(n_pts), inputs')        # n_pts × (1 + n_in)

    for d in axes(outputs, 1)
        y      = outputs[d, :]
        coeffs = A \ y
        y_pred = A * coeffs
        ss_res = sum((y .- y_pred).^2)
        ss_tot = sum((y .- mean(y)).^2)
        r2     = 1.0 - ss_res / max(ss_tot, 1e-15)

        terms = [string(round(coeffs[1]; sigdigits = 4))]
        for i in 1:n_in
            c = round(coeffs[i + 1]; sigdigits = 4)
            push!(terms, "$(c)·x$i")
        end
        println(io, "  output[$d] ≈ $(join(terms, " + "))   R²=$(round(r2; sigdigits = 3))")
    end
end

"""
    run_update_workflow(
        problem::AIRMEDProblem,
        drift_result::DriftResult,
        data_times, data_states;
        nn_hidden=16, nn_depth=2, kwargs...
    ) -> UpdateResult

Build a correction network and train it against the measurements, the update
step that follows a detected drift.

# Arguments
- `problem`: the problem being updated.
- `drift_result`: the detection that triggered the update.
- `base_ode!`, `nn_input_fn`, `u0`: the known physics, the network inputs and
  the initial state.
- `data_times`, `data_states`: the measurements.

# Keywords
- `nn_input_dim`, `nn_output_dim`, `nn_hidden`, `nn_depth`: network shape.
- `kwargs...`: forwarded to `train_ude`.

# Returns
The `UpdateResult` of the training.
"""
function run_update_workflow(
    problem::AIRMEDProblem,
    ::DriftResult,
    base_ode!::Function,
    nn_input_fn::Function,
    u0::AbstractVector,
    data_times::AbstractVector,
    data_states::AbstractMatrix;
    nn_input_dim::Int  = 2,
    nn_output_dim::Int = size(data_states, 1),
    nn_hidden::Int     = 16,
    nn_depth::Int      = 2,
    kwargs...,
)::UpdateResult
    nn = build_nn(nn_input_dim, nn_hidden, nn_output_dim; depth = nn_depth)
    return train_ude(base_ode!, nn, nn_input_fn, u0, problem.tspan,
                     data_times, data_states, problem.p0; kwargs...)
end
