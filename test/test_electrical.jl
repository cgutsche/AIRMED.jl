"""
Electrical circuit tests: RC circuit with unknown series/parallel resistors.
"""

using Test
using Statistics
using Random

include("electrical_fixture.jl")

@info "Running tests for: $(problem.name)"

@testset "Electrical Circuit — Base model vs. true data" begin

    @testset "Base model simulation runs" begin
        sol = simulate(problem)
        @test sol.retcode == ReturnCode.Success
        @test length(sol.t) > 1
        @test all(isfinite, Array(sol))
    end

    @testset "Drift is detected" begin
        saveat     = collect(range(T_SPAN...; length = 200))
        sol        = simulate(problem; saveat)
        sim_states = Array(sol)

        data_times, data_states = generate_true_data(T_SPAN, 200)

        _, residuals  = compute_residuals(sol.t, sim_states, data_times, data_states)
        scalar_res    = vec(mean(residuals; dims = 1))

        drift = detect_drift(scalar_res, problem.drift_config)
        @test drift.drift_detected
        @info "Drift detected at index $(drift.drift_index) using $(drift.method)"
    end

    @testset "UDE trains to lower loss than base model" begin
        data_times, data_states = generate_true_data(T_SPAN, 100)

        result = train_ude(
            base_ode!, build_nn(2, 16, 1; depth = 2), nn_input_fn,
            [0.0], T_SPAN, data_times, data_states, BASE_P;
            max_iters = 300,
            lr        = 1e-3,
            verbose   = false,
        )

        @info "UDE final loss: $(round(result.final_loss; sigdigits=4))"
        # Loss should decrease from the initial mismatch (~4 V² MSE)
        @test result.final_loss < 4.0
    end

    @testset "Symbolic UDE trains (ModelingToolkitNeuralNets tutorial approach)" begin
        data_times, data_states = generate_true_data(T_SPAN, 100)

        result = train_symbolic_ude(
            UDE_PROB, sym_nn_elec, θ_elec,
            data_times, data_states;
            max_iters = 500,
            lr        = 0.01,
            verbose   = false,
        )

        @info "Symbolic UDE final loss: $(round(result.final_loss; sigdigits=4))"
        @test result.success
        @test result.final_loss < 4.0
        @test !isnothing(result.fitted_ode_prob)
        @test !isnothing(result.sym_nn)

        # Verify the fitted function is callable (as in the tutorial)
        fitted_func(v_c, v_src) = result.fitted_ode_prob.ps[result.sym_nn](
            [v_c, v_src], result.fitted_ode_prob.ps[result.sym_theta])[1]
        val = fitted_func(2.5, V_SRC)
        @test isfinite(val)

        # Verify symbolic_regression_of_nn dispatches through UpdateResult
        inputs, outputs = symbolic_regression_of_nn(
            result, [(0.0, V_SRC), (V_SRC - 0.1, V_SRC + 0.1)]; n_samples = 10)
        @test size(inputs, 1) == 2
        @test size(outputs, 1) == 1
        @test all(isfinite, outputs)
    end

    @testset "Full run_airmed workflow" begin
        agent_state, update_result = run_airmed(
            problem, base_ode!, nn_input_fn;
            # This testset covers the UDE path, which `run_airmed` does not
            # take by default; without it `update_result` is `nothing`.
            train_ude    = true,
            n_points     = 100,
            verbose      = true,
            nn_input_dim = 2,
            nn_hidden    = 16,
            nn_depth     = 2,
            max_iters    = 300,
            lr           = 1e-3,
        )

        @test agent_state.n_iterations == 1
        @test agent_state.n_drift_events >= 1
        @test !isnothing(update_result)
        @test isfinite(update_result.final_loss)
        @test update_result.final_loss < 4.0
    end
end

# Independent of the circuit model above — exercises `DriftDetectionMethod`
# dispatch (types.jl) on synthetic residuals. Added alongside the CUSUM/EWMA/
# SimpleThreshold refactor so a future new method (or a dispatch regression
# that silently falls back to CUSUM) is caught here rather than only by
# manual inspection.
@testset "Drift Detection Methods" begin
    Random.seed!(20260908)

    # A clean baseline segment followed by a genuine, sustained level shift —
    # every method below must stay quiet on the first half and flag the second.
    residuals = vcat(0.1 .+ 0.02 .* randn(100), 0.6 .+ 0.02 .* randn(100))
    flat_only = 0.1 .+ 0.02 .* randn(200)   # negative control: no shift at all

    @testset "CUSUM" begin
        cfg = DriftConfig(method = CUSUM(k = 1.0, h = 15.0), min_samples = 30, min_sigma = 1e-4)
        r = detect_drift(residuals, cfg)
        @test r.drift_detected
        @test r.method == :cusum
        # Dispatch-correctness guard: these keys only exist if `_cusum_detect`
        # actually ran — a silent fallback to another method would show up here.
        @test haskey(r.details, :C_pos) && haskey(r.details, :C_neg)
        @test !detect_drift(flat_only, cfg).drift_detected
    end

    @testset "EWMA" begin
        cfg = DriftConfig(method = EWMA(lambda = 0.2, L = 3.0), min_samples = 30, min_sigma = 1e-4)
        r = detect_drift(residuals, cfg)
        @test r.drift_detected
        @test r.method == :ewma
        @test haskey(r.details, :ewma_stat) && haskey(r.details, :limit)
        @test !detect_drift(flat_only, cfg).drift_detected
    end

    @testset "SimpleThreshold" begin
        cfg = DriftConfig(method = SimpleThreshold(threshold = 0.3), min_samples = 30)
        r = detect_drift(residuals, cfg)
        @test r.drift_detected
        @test r.method == :threshold
        @test !isnothing(r.drift_index)
        @test !detect_drift(flat_only, cfg).drift_detected
    end

    @testset "default DriftConfig() matches pre-refactor default (CUSUM k=0.5 h=5.0)" begin
        @test DriftConfig().method isa CUSUM
        @test DriftConfig().method.k == 0.5
        @test DriftConfig().method.h == 5.0
    end

    @testset "n < min_samples short-circuits before dispatch" begin
        r = detect_drift(residuals[1:5], DriftConfig(method = CUSUM(), min_samples = 30))
        @test !r.drift_detected
        @test r.method == :cusum   # _method_symbol must still resolve pre-dispatch
    end

    @testset "_with_min_sigma / _with_scaled_threshold preserve method identity" begin
        base   = DriftConfig(method = CUSUM(k = 1.0, h = 15.0), min_samples = 30, min_sigma = 1e-4)
        raised = AIRMED._with_min_sigma(base, 1e-2)
        scaled = AIRMED._with_scaled_threshold(base, 0.5)
        @test raised.min_sigma == 1e-2
        @test raised.method.k == 1.0 && raised.method.h == 15.0   # untouched
        @test scaled.method.h == 7.5                              # 15.0 * 0.5
        @test scaled.method.k == 1.0                              # k does not scale
        @test scaled.min_sigma == 1e-4                            # untouched
    end
end
