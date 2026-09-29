"""
Hook-local characterisation tests. This step is arithmetic on the measured
channels, with no simulation, so each branch is driven by a synthetic signal
built for it: a resistive law, a switching load, an oscillation, a constant
sink, and nothing missing at all.

Relies on `problem`, the circuit components and the constants from
electrical_fixture.jl, which test_electrical.jl includes once — see the
ordering note in runtests.jl.
"""

using Test
using Random
using Statistics

const HL_OBS   = [cap.v, r1.v, r1.i, cap.i]
const HL_NOISE = [0.002, 0.002, 2e-5, 2e-5]
const HL_T     = collect(range(0.0, 0.05; length = 200))

# Measurement matrix for the RC circuit plus an extra current `extra(v, t)`
# drawn in parallel with R1, i.e. the balance-law residual at :r1_parallel.
function hl_data(extra; seed = 7)
    vc   = V_SRC .* (1 .- exp.(-HL_T ./ 0.03))
    vr1  = V_SRC .- vc
    ir1  = vr1 ./ R1_VAL
    icap = ir1 .+ extra(vr1, HL_T)
    X    = vcat(permutedims(vc), permutedims(vr1), permutedims(ir1), permutedims(icap))
    return X .+ HL_NOISE .* randn(Xoshiro(seed), size(X))
end

const PAR = HookResidualSpec(:r1_parallel, :parallel, r1.v, [cap.i => 1.0, r1.i => -1.0],
                             "extra path across R1")
const SER = HookResidualSpec(:r1_to_cap, :series, cap.i, [r1.v => 1.0, cap.v => 1.0],
                             "series drop")

hl_problem(specs = [PAR]; noise = HL_NOISE) = AIRMEDProblem(;
    name              = "RC — hook-local",
    model             = base_circuit,
    component_hooks   = problem.component_hooks,
    component_guesses = problem.component_guesses,
    data_source       = problem.data_source,
    tspan             = T_SPAN,
    u0                = [cap.v => 0.0],
    p0                = [r1.R => R1_VAL, cap.C => C_VAL, vref.k => V_SRC],
    observable_states = HL_OBS,
    observation_noise = noise,
    hook_residuals    = specs)

const X_RESISTOR = hl_data((v, t) -> v ./ R3_UNKNOWN)
const X_NOTHING  = hl_data((v, t) -> zero(v))

@testset "Hook-local characterisation" begin

    @testset "hook_snrs locates the missing element" begin
        @test AIRMED.hook_snrs(hl_problem(), HL_T, X_RESISTOR)[:r1_parallel] > 100
        @test AIRMED.hook_snrs(hl_problem(), HL_T, X_NOTHING)[:r1_parallel] < 3
        @test isempty(AIRMED.hook_snrs(problem, HL_T, X_RESISTOR))          # no residuals declared
        unmeasured = HookResidualSpec(:r1_parallel, :parallel, r1.v, [cap.p.i => 1.0])
        @test isempty(AIRMED.hook_snrs(hl_problem([unmeasured]), HL_T, X_RESISTOR))
    end

    @testset "a resistive law is characterised" begin
        s = AIRMED._characterise_hooks(hl_problem(), HL_T, X_RESISTOR)
        @test occursin("r1_parallel  [parallel] — extra path across R1", s)
        @test occursin("static-law check", s)
        @test occursin("sparse fit of the residual on the measured channels", s)
        @test occursin("response ≈", s)
        @test occursin("collinear", s)                     # cap.v = V_SRC - r1.v
        @test occursin("residual after the fit", s)
        @test occursin("identifiability: basis condition number", s)
        @test occursin("FORM NOT IDENTIFIABLE", s)         # operating point is always reported
        @test !occursin("x1", s)                           # channel names replace placeholders
    end

    @testset "a proportional law whose intercept is pruned is still significant" begin
        # response = drive / 2000 exactly, with an oscillating drive so that time
        # is not collinear with it. The fit recovers the single term, but the
        # significance test subtracts an intercept that was pruned, counts zero
        # terms and reports the element as [CONSTANT]. Known bug: see the F-test
        # in `_characterise_hooks`.
        drive = 5.0 .+ sin.(2π .* HL_T ./ 0.025)
        X = vcat(fill(1.0, 1, 200), permutedims(drive), fill(1e-3, 1, 200),
                 permutedims(1e-3 .+ drive ./ 2000))
        X = X .+ HL_NOISE .* randn(Xoshiro(11), size(X))
        s = AIRMED._characterise_hooks(hl_problem(), HL_T, X; terminal_inputs = true)
        @test occursin("response ≈", s)
        @test_broken !occursin("[CONSTANT]", s)
    end

    @testset "nothing missing: the residual sits at the noise floor" begin
        s = AIRMED._characterise_hooks(hl_problem(), HL_T, X_NOTHING)
        @test occursin("AT the noise floor; nothing is missing", s)
        @test !occursin("sparse fit", s)
    end

    @testset "include_analysis = false reports only the residual" begin
        s = AIRMED._characterise_hooks(hl_problem(), HL_T, X_RESISTOR; include_analysis = false)
        @test occursin("well above the noise floor", s)
        @test !occursin("sparse fit", s)
    end

    @testset "terminal_inputs restricts the fit to the element's own drive" begin
        s = AIRMED._characterise_hooks(hl_problem(), HL_T, X_RESISTOR; terminal_inputs = true)
        @test occursin("law in this element's OWN drive and time", s)
    end

    @testset "only_hooks restricts the report" begin
        s = AIRMED._characterise_hooks(hl_problem([PAR, SER]), HL_T, X_RESISTOR;
                                       only_hooks = [:r1_to_cap])
        @test occursin("r1_to_cap  [series]", s)
        @test !occursin("r1_parallel", s)
    end

    @testset "a switching load is described by its levels" begin
        square = hl_data((v, t) -> 1e-3 .* (mod.(t, 0.01) .< 0.005))
        s = AIRMED._characterise_hooks(hl_problem(), HL_T, square)
        @test occursin("SWITCHING", s)
        @test occursin("cycle period ≈", s)

        step = hl_data((v, t) -> 1e-3 .* (t .> 0.025))
        @test occursin("single switch — not periodic",
                       AIRMED._characterise_hooks(hl_problem(), HL_T, step))
    end

    @testset "an oscillation survives as a sin/cos pair" begin
        # Steady circuit, so time is the only varying input. During the RC
        # transient the near-collinear voltage channels inflate the least-squares
        # coefficients and the periodic pair is pruned against them.
        X = vcat(fill(2.0, 1, 200), fill(3.0, 1, 200), fill(3e-3, 1, 200),
                 permutedims(3e-3 .+ 5e-4 .* sin.(2π .* HL_T ./ 0.01)))
        X = X .+ HL_NOISE .* randn(Xoshiro(5), size(X))
        s = AIRMED._characterise_hooks(hl_problem(), HL_T, X)
        @test occursin("constant over this window (dropped)", s)
        @test occursin("OSCILLATES", s)
    end

    @testset "a constant sink depends on no channel" begin
        s = AIRMED._characterise_hooks(hl_problem(), HL_T, hl_data((v, t) -> fill(1e-3, length(t))))
        @test occursin("[CONSTANT]", s)
        @test occursin("consistent with a STATIC element", s)
        @test occursin("consistent with a static law", s)   # response differs only by noise
    end

    @testset "unmeasured channels are skipped, not guessed" begin
        no_drive = HookResidualSpec(:r1_parallel, :parallel, r1.p.v, [cap.i => 1.0])
        @test occursin("drive channel not among observable_states — skipped",
                       AIRMED._characterise_hooks(hl_problem([no_drive]), HL_T, X_RESISTOR))
        no_resp = HookResidualSpec(:r1_parallel, :parallel, r1.v, [cap.p.i => 1.0])
        @test occursin("a response channel is not among observable_states — skipped",
                       AIRMED._characterise_hooks(hl_problem([no_resp]), HL_T, X_RESISTOR))
        self_ref = HookResidualSpec(:r1_parallel, :parallel, r1.v, [r1.v => 1.0, cap.i => 1.0])
        @test_logs (:warn, r"drive is also a response term") match_mode = :any AIRMED._characterise_hooks(
            hl_problem([self_ref]), HL_T, X_RESISTOR)
        @test AIRMED._characterise_hooks(problem, HL_T, X_RESISTOR) == ""   # nothing declared
    end

    @testset "_hl_mode_analysis" begin
        σ = 1e-5
        sq = 1e-3 .* (mod.(HL_T, 0.01) .< 0.005)
        m = AIRMED._hl_mode_analysis(HL_T, sq, σ)
        @test m.hi_lvl - m.lo_lvl ≈ 1e-3
        @test isapprox(m.duty, 0.5; atol = 0.05)
        @test 8 <= m.transitions <= 10
        @test isapprox(m.period, 0.01; rtol = 0.15)
        @test isnothing(AIRMED._hl_mode_analysis(HL_T, σ .* randn(Xoshiro(1), 200), σ))  # one noisy mode
        @test isnothing(AIRMED._hl_mode_analysis(HL_T, fill(1.0, 200), σ))
        @test isnothing(AIRMED._hl_mode_analysis(HL_T[1:5], sq[1:5], σ))                # too short
    end

    @testset "_hl_dominant_period" begin
        @test isapprox(AIRMED._hl_dominant_period(HL_T, sin.(2π .* HL_T ./ 0.01)), 0.01; rtol = 0.02)
        @test isnan(AIRMED._hl_dominant_period(HL_T, zeros(200)))
        @test isnan(AIRMED._hl_dominant_period(HL_T[1:20], sin.(HL_T[1:20])))
    end

    @testset "_hl_static_test separates a static law from hidden state" begin
        # Two periods: every input value recurs 100 samples later, well beyond
        # `min_gap`, so a static law gives identical responses at those pairs.
        x = sin.(2π .* (0:199) ./ 100)
        X = permutedims(x)
        frac, mem = AIRMED._hl_static_test(X, 2 .* x, 1e-3)
        @test mem < 1e-9 && frac ≈ 1.0
        _, mem_h = AIRMED._hl_static_test(X, x .+ (1:200) ./ 100, 1e-3)   # also depends on time
        @test mem_h > 0.1
        @test all(isnan, AIRMED._hl_static_test(X[:, 1:5], x[1:5], 1e-3))   # no admissible pair
    end

    @testset "_hl_basis_predict evaluates a fit on its own basis" begin
        X = rand(Xoshiro(3), 2, 50)
        y = 1.0 .+ 2.0 .* X[1, :]
        names, Φ = AIRMED._build_basis(X, 1, Pair{String, Function}[])
        fit = first(sparse_regression(X, permutedims(y); basis_degree = 1))
        @test AIRMED._hl_basis_predict(fit, names, Φ) ≈ y
        @test_throws ErrorException AIRMED._hl_basis_predict((; terms = ["nope" => 1.0]), names, Φ)
    end

    @testset "_component_law_lines" begin
        laws = AIRMED._component_law_lines(problem.component_guesses[1])   # Resistor
        @test !isempty(laws)
        @test !any(l -> occursin("p₊", l) || occursin("n₊", l), laws)      # pin boilerplate dropped
        broken = ComponentGuess(:Broken, "d", (n, v) -> error("no such component"))
        @test isempty(AIRMED._component_law_lines(broken))
    end

    @testset "fitted structures are judged against the declared sensor noise" begin
        # Physically consistent data: the true circuit is the base model with a
        # resistor across R1, simulated and read out on all four sensors.
        truth = ComponentProposal(:r3, :Resistor, (:r1, :p), (:r1, :n), Dict(:R => R3_UNKNOWN), "")
        true_prob, _, _ = build_adapted_problem(hl_problem(), [truth])
        sol = simulate(true_prob; saveat = HL_T)
        X = reduce(vcat, [permutedims(Float64.(sol[s])) for s in HL_OBS])
        X = X .+ HL_NOISE .* randn(Xoshiro(21), size(X))
        # Demo attempt 1 is a resistor at r1_parallel: the correct structure.
        kw = (; data_times = HL_T, data_states = X, api = :none, max_retries = 0, fit_iters = 100)

        good = propose_model_adaptation(hl_problem(); kw...)
        @test good.success
        @test occursin("structure quality: good", good.message)
        @test isapprox(only(good.proposals).parameters[:R], R3_UNKNOWN; rtol = 0.05)

        # Declaring the sensors 10x cleaner than they are puts every channel far
        # above its floor, so the same correct structure is rejected.
        strict = propose_model_adaptation(hl_problem(; noise = HL_NOISE ./ 10); kw...,
                                          escalate_on_inadequate_structure = true)
        @test !strict.success
        @test occursin("REJECTED", strict.message)
        @test occursin("x sensor noise", strict.message)
        @test !isempty(strict.escalation)
    end

    @testset "the hook-local evidence reaches the adaptation prompt" begin
        kw = (; data_times = HL_T, api = :none, max_retries = 0, fit_params = false)
        res = propose_model_adaptation(hl_problem(); data_states = X_RESISTOR, kw...,
                                       restrict_hooks_by_snr = true,
                                       include_component_equations = true)
        @test occursin("## Hook-local characterisation (measured, not inferred)", res.prompt)
        @test occursin("free parameter: R", res.prompt)
        @test occursin("- r1_parallel:", res.prompt)
        @test !occursin("- r1_to_cap:", res.prompt)       # below the SNR threshold, so not offered

        res = @test_logs (:warn, r"no hook clears SNR") match_mode = :any propose_model_adaptation(
            hl_problem(); data_states = X_NOTHING, kw..., restrict_hooks_by_snr = true)
        @test occursin("- r1_to_cap:", res.prompt)        # fell back to the full list

        res = propose_model_adaptation(hl_problem(); data_states = X_RESISTOR, kw...,
                                       hook_local_analysis = false)
        @test occursin("well above the noise floor", res.prompt)

        res = propose_model_adaptation(hl_problem(); data_states = X_RESISTOR, kw...,
                                       hook_summary = "PRECOMPUTED HOOK SUMMARY")
        @test occursin("PRECOMPUTED HOOK SUMMARY", res.prompt)
    end
end
