"""
`run_airmed` branches beyond the UDE-only path in test_electrical.jl: the
structural adaptation with and without a UDE, the drift-explained gate, and
escalation. UDE training uses `max_iters = 0` (an untrained network with valid
diagnostics), since these tests concern the control flow, not learning.

Relies on `problem`, `base_ode!` and `nn_input_fn` from electrical_fixture.jl,
which test_electrical.jl includes once — see the ordering note in runtests.jl.
"""

using Test

const CHEAP_ADAPT = (; max_retries = 0, fit_iters = 30)

@testset "run_airmed workflow branches" begin

    @testset "adapt = true without a UDE runs the structural adaptation" begin
        state, upd, adaptation = run_airmed(problem, base_ode!, nn_input_fn;
                                            n_points = 100, verbose = true, adapt = true,
                                            adapt_kwargs = CHEAP_ADAPT)
        @test isnothing(upd)
        @test !isnothing(adaptation)
        @test adaptation.api == :none                 # taken from supervision_config
        @test length(state.adaptation_history) == 1
    end

    @testset "a UDE that explains too little of the drift gates the adaptation" begin
        state, upd, adaptation = @test_logs (:warn, r"skipping structural adaptation") match_mode = :any run_airmed(
            problem, base_ode!, nn_input_fn; n_points = 100, verbose = false,
            train_ude = true, max_iters = 0, adapt = true,
            adapt_min_drift_explained = 1.0, adapt_kwargs = CHEAP_ADAPT)
        @test !isnothing(upd) && upd.success
        @test isnothing(adaptation)
        @test length(state.update_history) == 1
    end

    @testset "with the gate disabled, the UDE evidence reaches the prompt" begin
        _, upd, adaptation = run_airmed(problem, base_ode!, nn_input_fn;
                                        n_points = 100, verbose = false,
                                        train_ude = true, max_iters = 0, adapt = true,
                                        adapt_min_drift_explained = 0.0,
                                        adapt_kwargs = CHEAP_ADAPT)
        @test !isnothing(adaptation)
        # Input ranges were derived from the fitted trajectory, so the network
        # was characterised rather than skipped.
        @test occursin("Temporal signature analysis", adaptation.prompt)
        @test !isnothing(AIRMED._auto_input_ranges(upd))
    end

    @testset "escalation skips training and adaptation" begin
        esc = rebuild_problem(problem; supervision_config = SupervisionConfig(
            ask_human_on_uncertainty = true, human_threshold = 1e-6))
        state, upd, adaptation = @test_logs (:warn, r"Escalating to human") match_mode = :any run_airmed(
            esc, base_ode!, nn_input_fn; n_points = 100, verbose = false,
            train_ude = true, adapt = true)
        @test state.escalated
        @test isnothing(upd) && isnothing(adaptation)
    end

    @testset "state accumulates across calls" begin
        state, _, _ = run_airmed(problem, base_ode!, nn_input_fn; n_points = 100, verbose = false)
        state, _, _ = run_airmed(problem, base_ode!, nn_input_fn; n_points = 100, verbose = false, state)
        @test state.n_iterations == 2
        @test state.n_drift_events == 2
    end
end
