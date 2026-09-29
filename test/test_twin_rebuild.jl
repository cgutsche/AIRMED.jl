"""
Twin-rebuild tests: turning a `ModelAdaptationResult` (or a manually built
`ComponentProposal` / `ParameterAdjustment`) into a runnable `AIRMEDProblem`.

Relies on `problem`, `r1`, `R1_VAL` and the electrical-circuit constants
already defined by electrical_fixture.jl, which test_electrical.jl includes
once — see the ordering note in runtests.jl.
"""

using Test
using ModelingToolkit

@testset "Twin Rebuild" begin

    manual_proposal() = ComponentProposal(:r_extra, :Resistor,
        problem.component_hooks[1].port_a, problem.component_hooks[1].port_b,
        Dict(:R => 2000.0), "manual test proposal")

    @testset "build_adapted_system inserts a component at a hook" begin
        sys, guesses, msg, comps = build_adapted_system(problem, [manual_proposal()])
        @test !isnothing(sys)
        @test !isnothing(comps)
        @test length(comps) == 1
        @test guesses isa Dict
    end

    @testset "build_adapted_system reports a proposal with no matching ComponentGuess" begin
        bad = ComponentProposal(:x, :Transformer,
            problem.component_hooks[1].port_a, problem.component_hooks[1].port_b,
            Dict{Symbol, Float64}(), "no such guess")
        sys, _, msg, comps = build_adapted_system(problem, [bad])
        @test isnothing(sys)
        @test isnothing(comps)
        @test occursin("no ComponentGuess", msg)
    end

    @testset "build_adapted_problem returns a simulatable AIRMEDProblem" begin
        adapted, msg, comps = build_adapted_problem(problem, [manual_proposal()])
        @test !isnothing(adapted)
        @test !isnothing(comps)
        # The search space is cleared on the rebuilt twin: it is for
        # simulation, not further adaptation.
        @test isempty(adapted.component_hooks)
        @test isempty(adapted.adaptable_params)
        @test isempty(adapted.hook_residuals)
        @test isempty(adapted.observation_noise)

        sol = simulate(adapted)
        @test sol.retcode == ReturnCode.Success
        @test all(isfinite, Array(sol))
    end

    @testset "build_adapted_problem: no proposals" begin
        prob, msg, comps = build_adapted_problem(problem, ComponentProposal[])
        @test isnothing(prob)
        @test isnothing(comps)
        @test occursin("no proposals", msg)
    end

    @testset "rebuild_problem overrides only the named fields" begin
        renamed = rebuild_problem(problem; name = "renamed twin")
        @test renamed.name == "renamed twin"
        @test renamed.model === problem.model
        @test renamed.tspan == problem.tspan
        @test renamed.component_hooks == problem.component_hooks
        # `==` on symbolic MTK terms builds a symbolic predicate, not a Bool;
        # `isequal` is the structural comparison used elsewhere in the codebase.
        @test isequal(renamed.observable_states, problem.observable_states)
    end

    @testset "build_parameter_adjusted_problem replaces the matching p0 entry" begin
        adj = ParameterAdjustment(:r1_R, r1.R, R1_VAL, 1.5 * R1_VAL, "manual test")
        adapted, msg = build_parameter_adjusted_problem(problem, [adj])
        @test occursin("1 parameter", msg)
        new_r1 = first(v for (k, v) in adapted.p0 if isequal(k, r1.R))
        @test new_r1 == 1.5 * R1_VAL
        # Search space cleared here too, same as the structural path.
        @test isempty(adapted.component_hooks)
        @test isempty(adapted.adaptable_params)
    end

    @testset "build_parameter_adjusted_problem reports a parameter missing from p0" begin
        @parameters nonexistent_param
        adj = ParameterAdjustment(:ghost, nonexistent_param, 1.0, 2.0, "not in p0")
        _, msg = build_parameter_adjusted_problem(problem, [adj])
        @test occursin("skipped", msg)
        @test occursin("ghost", msg) || occursin("nonexistent_param", msg)
    end

    @testset "build_fix_problem dispatches on parameter vs structural outcome" begin
        adj_result = ModelAdaptationResult(
            ComponentProposal[], [ParameterAdjustment(:r1_R, r1.R, R1_VAL, 1.2 * R1_VAL, "")],
            "", nothing, "", "", :none, true, "parameter fix", 0.0, LLMExchange[], "")
        prob1, msg1, comps1 = build_fix_problem(problem, adj_result)
        @test !isnothing(prob1)
        @test isnothing(comps1)   # no component is created for a parameter fix

        struct_result = ModelAdaptationResult(
            [manual_proposal()], ParameterAdjustment[],
            "", nothing, "", "", :none, true, "structural fix", 0.0, LLMExchange[], "")
        prob2, msg2, comps2 = build_fix_problem(problem, struct_result)
        @test !isnothing(prob2)
        @test !isnothing(comps2)
        @test length(comps2) == 1

        empty_result = ModelAdaptationResult(
            ComponentProposal[], ParameterAdjustment[],
            "", nothing, "", "", :none, false, "nothing found", NaN, LLMExchange[], "")
        prob3, msg3, comps3 = build_fix_problem(problem, empty_result)
        @test isnothing(prob3)
        @test isnothing(comps3)
        @test occursin("neither", msg3)
    end
end
