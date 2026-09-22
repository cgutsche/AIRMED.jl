"""
Continuous model validation: residual computation and drift detection.
Methods: CUSUM, EWMA, simple threshold.
"""

using Statistics
using LinearAlgebra

"""
    compute_residuals(sim_times, sim_states, data_times, data_states) -> (times, residuals)

Interpolate the simulation to the measurement times and compare the two.

# Arguments
- `sim_times`, `sim_states`: the simulated trajectory.
- `data_times`, `data_states`: the measurements. Both state arguments are
  `(n_states x n_points)` matrices.

# Returns
`(times, residuals)`, the measurement times and the point-wise absolute
difference.
"""
function compute_residuals(
    sim_times::AbstractVector,
    sim_states::AbstractMatrix,
    data_times::AbstractVector,
    data_states::AbstractMatrix,
)
    n_states = size(data_states, 1)
    n_data   = length(data_times)
    interp   = Matrix{Float64}(undef, n_states, n_data)

    for (j, t) in enumerate(data_times)
        idx = searchsortedlast(sim_times, t)
        idx = clamp(idx, 1, length(sim_times) - 1)
        t0, t1 = sim_times[idx], sim_times[idx + 1]
        α = t1 == t0 ? 0.0 : (t - t0) / (t1 - t0)
        interp[:, j] = (1 - α) .* sim_states[:, idx] .+ α .* sim_states[:, idx + 1]
    end

    residuals = abs.(data_states .- interp)
    return data_times, residuals
end

"""
    detect_drift(residuals::AbstractMatrix, config::DriftConfig) -> DriftResult

Run drift detection on an `(n_states x n_times)` residual matrix, averaged
across states.

# Arguments
- `residuals`: the residual matrix.
- `config`: the detection method and its baseline settings.

# Returns
A `DriftResult` for the averaged series.
"""
function detect_drift(residuals::AbstractMatrix, config::DriftConfig)::DriftResult
    scalar_residuals = vec(mean(abs.(residuals), dims=1))
    return detect_drift(scalar_residuals, config)
end

"""
    detect_drift(residuals::AbstractVector, config::DriftConfig) -> DriftResult

Run drift detection on a scalar residual series.

# Arguments
- `residuals`: the series.
- `config`: the detection method and its baseline settings.

# Returns
A `DriftResult` with the verdict, the detection index and the method's own
statistics.
"""
function detect_drift(residuals::AbstractVector{<:Real}, config::DriftConfig)::DriftResult
    n = length(residuals)
    if n < config.min_samples
        return DriftResult(false, nothing, mean(residuals), maximum(residuals),
                           _method_symbol(config.method), Dict{Symbol,Any}())
    end
    return _detect(residuals, config, config.method)
end

# `DriftResult.method` is a display and comparison field, independent of the
# `DriftDetectionMethod` type hierarchy. A new subtype needs one line here.
_method_symbol(::CUSUM)           = :cusum
_method_symbol(::EWMA)            = :ewma
_method_symbol(::SimpleThreshold) = :threshold

# Dispatch on the concrete method type: a new algorithm needs a subtype in
# types.jl and one `_detect` method here, leaving `detect_drift` unchanged.
_detect(residuals, config, m::CUSUM)           = _cusum_detect(residuals, config, m)
_detect(residuals, config, m::EWMA)            = _ewma_detect(residuals, config, m)
_detect(residuals, config, m::SimpleThreshold) = _threshold_detect(residuals, config, m)

function _threshold_detect(residuals, config, m::SimpleThreshold)
    above   = residuals .> m.threshold
    idx     = findfirst(above)
    detected = !isnothing(idx)
    return DriftResult(detected, idx, mean(residuals), maximum(residuals),
                       :threshold, Dict{Symbol,Any}(:threshold => m.threshold))
end

function _cusum_detect(residuals, config, m::CUSUM)
    n    = length(residuals)
    μ    = mean(residuals[1:config.min_samples])
    σ    = std(residuals[1:config.min_samples]; corrected=false)
    σ    = max(σ, config.min_sigma, 1e-12)
    k, h = m.k, m.h

    C_pos = zeros(n)
    C_neg = zeros(n)

    for i in 2:n
        z         = (residuals[i] - μ) / σ
        C_pos[i]  = max(0.0, C_pos[i-1] + z - k)
        C_neg[i]  = max(0.0, C_neg[i-1] - z - k)
    end

    combined = max.(C_pos, C_neg)
    detected = any(combined .> h)
    idx      = findfirst(>(h), combined)

    return DriftResult(detected, idx, mean(residuals), maximum(residuals), :cusum,
                       Dict{Symbol,Any}(:C_pos => C_pos, :C_neg => C_neg, :h => h))
end

function _ewma_detect(residuals, config, m::EWMA)
    n      = length(residuals)
    λ      = m.lambda
    L      = m.L
    μ      = mean(residuals[1:config.min_samples])
    σ      = std(residuals[1:config.min_samples]; corrected=false)
    σ      = max(σ, config.min_sigma, 1e-12)
    limit  = L * σ * sqrt(λ / (2 - λ))

    z    = zeros(n)
    z[1] = residuals[1]
    for i in 2:n
        z[i] = λ * residuals[i] + (1 - λ) * z[i-1]
    end

    deviations = abs.(z .- μ)
    detected   = any(deviations .> limit)
    idx        = findfirst(>(limit), deviations)

    return DriftResult(detected, idx, mean(residuals), maximum(residuals), :ewma,
                       Dict{Symbol,Any}(:ewma_stat => z, :limit => limit))
end
