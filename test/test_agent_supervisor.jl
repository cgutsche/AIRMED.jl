"""
Agent-supervisor tests: AgentState bookkeeping and the deterministic decision
paths of `agent_supervise!`. The LLM-backed decision path
(`SupervisionConfig(use_agent = true, agent_api = ...)`) needs a real backend
and is exercised by the evaluation repository instead — every path tested
here uses `use_agent = false`, so no network access or API key is required.
"""

using Test

mk_drift(detected::Bool; max_residual::Real = 0.1) =
    DriftResult(detected, detected ? 1 : nothing, 0.05, max_residual, :cusum, Dict{Symbol, Any}())

@testset "Agent Supervisor" begin

    @testset "AgentState starts empty" begin
        st = AgentState()
        @test st.n_iterations == 0
        @test st.n_drift_events == 0
        @test st.n_updates == 0
        @test !st.escalated
        @test isempty(st.residual_history)
        @test isempty(st.drift_history)
        @test isempty(st.update_history)
        @test isempty(st.adaptation_history)
        @test isempty(st.log)
    end

    @testset ":continue when no drift is detected" begin
        st = AgentState()
        decision = agent_supervise!(st, [0.01, 0.02], mk_drift(false), SupervisionConfig())
        @test decision == :continue
        @test st.n_iterations == 1
        @test st.n_drift_events == 0
        @test length(st.drift_history) == 1
        @test length(st.log) == 1
    end

    @testset ":update on an ordinary drift with no agent and thresholds not tripped" begin
        st  = AgentState()
        cfg = SupervisionConfig(use_agent = false, escalate_on_repeated_drift = false,
                                ask_human_on_uncertainty = false)
        decision = agent_supervise!(st, [0.3, 0.4], mk_drift(true; max_residual = 0.2), cfg)
        @test decision == :update
        @test st.n_drift_events == 1
        @test !st.escalated
    end

    @testset ":escalate once repeated drift exceeds max_auto_updates" begin
        st  = AgentState()
        cfg = SupervisionConfig(use_agent = false, escalate_on_repeated_drift = true,
                                max_auto_updates = 2, ask_human_on_uncertainty = false)
        decisions = [agent_supervise!(st, [0.3], mk_drift(true; max_residual = 0.2), cfg) for _ in 1:3]
        @test decisions == [:update, :update, :escalate]
        @test st.n_drift_events == 3
        @test st.escalated
    end

    @testset ":escalate when a single residual exceeds human_threshold" begin
        st  = AgentState()
        cfg = SupervisionConfig(use_agent = false, escalate_on_repeated_drift = false,
                                ask_human_on_uncertainty = true, human_threshold = 0.5)
        decision = agent_supervise!(st, [0.9], mk_drift(true; max_residual = 0.9), cfg)
        @test decision == :escalate
        @test st.escalated
    end

    @testset "escalation conditions are checked before the agent, and cannot be overridden by it" begin
        # use_agent = true but agent_api = :none — if the agent were consulted
        # it would error or hang; the repeated-drift escalation must fire
        # first and never reach that call.
        st  = AgentState()
        cfg = SupervisionConfig(use_agent = true, agent_api = :none,
                                escalate_on_repeated_drift = true, max_auto_updates = 0,
                                ask_human_on_uncertainty = false)
        decision = agent_supervise!(st, [0.3], mk_drift(true; max_residual = 0.2), cfg)
        @test decision == :escalate
    end

    @testset "use_agent with agent_api = :none defaults to :update" begin
        st  = AgentState()
        cfg = SupervisionConfig(use_agent = true, agent_api = :none,
                                escalate_on_repeated_drift = false, ask_human_on_uncertainty = false)
        decision = agent_supervise!(st, [0.3], mk_drift(true; max_residual = 0.2), cfg)
        @test decision == :update
    end

    @testset "record_update! counts only successful updates" begin
        st  = AgentState()
        ok  = UpdateResult(true, 0.01, 50, nothing, nothing, nothing, "ok")
        bad = UpdateResult(false, NaN, 0, nothing, nothing, nothing, "failed")
        record_update!(st, ok)
        record_update!(st, bad)
        @test length(st.update_history) == 2
        @test st.n_updates == 1
    end

    @testset "record_adaptation! appends regardless of outcome" begin
        st = AgentState()
        adapt = ModelAdaptationResult(
            ComponentProposal[], ParameterAdjustment[], "", nothing, "", "",
            :none, true, "no-op adaptation", NaN, LLMExchange[], "")
        record_adaptation!(st, adapt)
        @test length(st.adaptation_history) == 1
    end

    @testset "generate_report summarises the accumulated state" begin
        st = AgentState()
        agent_supervise!(st, [0.01], mk_drift(false), SupervisionConfig())
        record_update!(st, UpdateResult(true, 0.01, 10, nothing, nothing, nothing, "ok"))
        report = generate_report(st)
        @test report isa String
        @test occursin("Iterations:", report)
        @test occursin("Model updates:", report)
        @test occursin("Escalated:", report)
        @test occursin("--- Log ---", report)
    end

    @testset "explain_for_user with no backend configured" begin
        txt = explain_for_user("explain this finding", SupervisionConfig())   # agent_api = :none
        @test occursin("no LLM configured", txt)
    end
end
