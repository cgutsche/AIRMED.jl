"""
Type-construction validation: the error paths of the `AdaptableParam`,
`ComponentGuess`, `HookResidualSpec` and `AIRMEDProblem` keyword constructors.

Relies on `base_circuit`, `r1`, `cap`, `vref`, `problem` and the electrical
constants already defined by electrical_fixture.jl, which test_electrical.jl
includes once — see the ordering note in runtests.jl.
"""

using Test
using ModelingToolkitStandardLibrary.Electrical: Resistor

@testset "Type construction validation" begin

    @testset "AdaptableParam bounds" begin
        @test_throws ArgumentError AdaptableParam(:r1_R, r1.R, R1_VAL; param_min = -1.0)
        @test_throws ArgumentError AdaptableParam(:r1_R, r1.R, R1_VAL; param_min = 10.0, param_max = 5.0)
        ap = AdaptableParam(:r1_R, r1.R, R1_VAL)
        @test ap.param_min == 1e-12
        @test ap.param_max == 1e12
    end

    @testset "ComponentGuess bounds" begin
        factory = (name, val) -> Resistor(; R = val, name = name)
        @test_throws ArgumentError ComponentGuess(:Resistor, "d", factory; param_min = 0.0)
        @test_throws ArgumentError ComponentGuess(:Resistor, "d", factory; param_min = 10.0, param_max = 1.0)
        g = ComponentGuess(:Resistor, "d", factory)
        @test g.param_min == 1e-12 && g.param_max == 1e12
        @test g.connector_names == (:p, :n)
    end

    @testset "HookResidualSpec kind validation" begin
        @test_throws ArgumentError HookResidualSpec(:h, :sideways, cap.v, [cap.v => 1.0])
        spec = HookResidualSpec(:h, :parallel, cap.v, [cap.v => 1.0])
        @test spec.kind == :parallel
        # `==` on symbolic MTK terms builds a symbolic predicate, not a Bool.
        @test isequal(spec.response_terms, [cap.v => 1.0])
    end

    @testset "AIRMEDProblem: hook_residuals referencing an unknown hook" begin
        bad_spec = HookResidualSpec(:no_such_hook, :parallel, cap.v, [cap.v => 1.0])
        @test_throws ArgumentError AIRMEDProblem(;
            name              = "bad",
            model             = base_circuit,
            component_hooks   = problem.component_hooks,
            hook_residuals    = [bad_spec],
            observable_states = [cap.v],
            observation_noise = [0.01],
            data_source       = problem.data_source,
            tspan             = T_SPAN,
            u0                = [cap.v => 0.0],
            p0                = [r1.R => R1_VAL, cap.C => C_VAL, vref.k => V_SRC])
    end

    @testset "AIRMEDProblem: observation_noise length mismatch" begin
        good_spec = HookResidualSpec(problem.component_hooks[1].name, :parallel, cap.v, [cap.v => 1.0])
        @test_throws ArgumentError AIRMEDProblem(;
            name              = "bad",
            model             = base_circuit,
            component_hooks   = problem.component_hooks,
            hook_residuals    = [good_spec],
            observable_states = [cap.v],
            observation_noise = [0.01, 0.02],   # 2 entries, 1 observable state
            data_source       = problem.data_source,
            tspan             = T_SPAN,
            u0                = [cap.v => 0.0],
            p0                = [r1.R => R1_VAL, cap.C => C_VAL, vref.k => V_SRC])
    end

    @testset "AIRMEDProblem: hook_residuals without observable_states" begin
        good_spec = HookResidualSpec(problem.component_hooks[1].name, :parallel, cap.v, [cap.v => 1.0])
        @test_throws ArgumentError AIRMEDProblem(;
            name            = "bad",
            model           = base_circuit,
            component_hooks = problem.component_hooks,
            hook_residuals  = [good_spec],
            data_source     = problem.data_source,
            tspan           = T_SPAN,
            u0              = [cap.v => 0.0],
            p0              = [r1.R => R1_VAL, cap.C => C_VAL, vref.k => V_SRC])
    end

    @testset "ModelAdaptationResult: backward-compatible constructors fill defaults" begin
        r9 = ModelAdaptationResult(ComponentProposal[], "code", nothing, "p", "r", :none, true, "m", 0.5)
        @test isempty(r9.parameter_adjustments) && isempty(r9.conversation)
        @test r9.fit_loss == 0.5 && r9.escalation == ""
        r8 = ModelAdaptationResult(ComponentProposal[], "code", nothing, "p", "r", :none, true, "m")
        @test isnan(r8.fit_loss)
    end

    @testset "AIRMEDProblem: valid construction with empty optional fields" begin
        p = AIRMEDProblem(;
            name        = "minimal",
            model       = base_circuit,
            data_source = problem.data_source,
            tspan       = T_SPAN,
            u0          = [cap.v => 0.0],
            p0          = [r1.R => R1_VAL, cap.C => C_VAL, vref.k => V_SRC])
        @test isempty(p.component_hooks)
        @test isempty(p.adaptable_params)
        @test isempty(p.optimizable_params)
        @test isempty(p.hook_residuals)
    end
end
