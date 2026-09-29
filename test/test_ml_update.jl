"""
Symbolic-regression and update-summary tests, plus the small `run_airmed`
helpers. Every `UpdateResult` here is built from an untrained network, so no
training runs: these test the bookkeeping and sampling code, not learning.

Relies on `problem` and the constants from electrical_fixture.jl, which
test_electrical.jl includes once — see the ordering note in runtests.jl.
"""

using Test
using Random
using Lux

# A Lux-path UpdateResult with a fixed trajectory: input 1 sweeps 0..1, input 2
# is held at 5.0, as a source voltage would be.
function synthetic_update_result(; success = true, metrics = Dict{Symbol, Any}())
    nn     = build_nn(2, 4, 1; depth = 1)
    ps, st = Lux.setup(Xoshiro(0), nn)
    traj   = vcat(permutedims(collect(range(0.0, 1.0; length = 20))), fill(5.0, 1, 20))
    return UpdateResult(success, 0.01, 10, nn, ps, st, "synthetic",
                        nothing, nothing, nothing, traj, metrics)
end

@testset "ML update helpers" begin

    @testset "sparse_regression recovers a sparse polynomial" begin
        rng = Xoshiro(42)
        x   = rand(rng, 2, 200) .* 2 .- 1
        y   = 2.0 .+ 3.0 .* x[1, :] .- 0.5 .* x[1, :] .* x[2, :]
        fit = only(sparse_regression(x, permutedims(y)))
        terms = Dict(fit.terms)
        @test fit.r2 > 0.999999
        @test isapprox(terms["1"], 2.0; atol = 1e-8)
        @test isapprox(terms["x1"], 3.0; atol = 1e-8)
        @test isapprox(terms["x1*x2"], -0.5; atol = 1e-8)
        @test length(terms) == 3                      # everything else pruned
        @test occursin("x1*x2", fit.expression)
    end

    @testset "sparse_regression: one result per output channel" begin
        x = rand(Xoshiro(1), 1, 50)
        results = sparse_regression(x, vcat(2 .* x, 3 .* x .^ 2); basis_degree = 2)
        @test length(results) == 2
        @test isapprox(Dict(results[1].terms)["x1"], 2.0; atol = 1e-8)
        @test isapprox(Dict(results[2].terms)["x1^2"], 3.0; atol = 1e-8)
    end

    @testset "sparse_regression: caller-supplied basis terms" begin
        x   = rand(Xoshiro(2), 1, 100) .* 3
        y   = permutedims(sin.(x[1, :]))
        fit = only(sparse_regression(x, y; basis_degree = 1,
                                     extra_basis = ["sin(x1)" => v -> sin(v[1])]))
        @test isapprox(Dict(fit.terms)["sin(x1)"], 1.0; atol = 1e-8)
        @test fit.r2 > 0.999999
    end

    @testset "sparse_regression: a failing basis term is skipped with a warning" begin
        x = rand(Xoshiro(3), 1, 30)
        results = @test_logs (:warn, r"extra basis term \"bad\" failed") match_mode = :any sparse_regression(
            x, 2 .* x; basis_degree = 1, extra_basis = ["bad" => v -> error("boom")])
        @test !any(n == "bad" for (n, _) in only(results).terms)
    end

    @testset "sparse_regression: scale_columns returns coefficients in original units" begin
        rng = Xoshiro(4)
        x   = vcat(rand(rng, 1, 100) .* 1e3, rand(rng, 1, 100) .* 1e-3)
        y   = permutedims(2e-3 .* x[1, :] .+ 5e2 .* x[2, :])
        fit = only(sparse_regression(x, y; basis_degree = 1, scale_columns = true))
        @test isapprox(Dict(fit.terms)["x1"], 2e-3; rtol = 1e-6)
        @test isapprox(Dict(fit.terms)["x2"], 5e2; rtol = 1e-6)
    end

    @testset "sparse_regression: an all-zero output gives the expression \"0\"" begin
        fit = only(sparse_regression(rand(Xoshiro(5), 1, 20), zeros(1, 20); basis_degree = 1))
        @test fit.expression == "0"
        @test isempty(fit.terms)
    end

    @testset "symbolic_regression_of_nn: trajectory sampling stays within the ranges" begin
        ur     = synthetic_update_result()
        ranges = [(0.0, 1.0), (4.9, 5.1)]
        inputs, outputs = symbolic_regression_of_nn(ur, ranges)
        @test size(inputs) == (2, 60)                 # trajectory + two jittered copies
        @test size(outputs) == (1, 60)
        @test all(0.0 .<= inputs[1, :] .<= 1.0)
        @test all(4.9 .<= inputs[2, :] .<= 5.1)
        @test all(isfinite, outputs)
    end

    @testset "symbolic_regression_of_nn: grid fallback" begin
        ur = synthetic_update_result()
        inputs, outputs = symbolic_regression_of_nn(ur, [(0.0, 1.0), (4.0, 6.0)];
                                                    use_trajectory = false, n_samples = 5)
        @test size(inputs) == (2, 25)
        @test size(outputs) == (1, 25)
        # The grid is capped by `max_grid_points`, never below 2 points per dimension.
        inputs2, _ = symbolic_regression_of_nn(ur, [(0.0, 1.0), (4.0, 6.0)];
                                               use_trajectory = false, max_grid_points = 1)
        @test size(inputs2, 2) == 4
    end

    @testset "_eval_nn_at on the Lux path" begin
        y = AIRMED._eval_nn_at(synthetic_update_result(), [0.5, 5.0])
        @test length(y) == 1
        @test all(isfinite, y)
    end

    @testset "model_update_summary" begin
        metrics = Dict{Symbol, Any}(:per_state_rmse_base => [0.1, 0.2],
                                    :per_state_rmse_ude  => [0.01, 0.2],
                                    :drift_explained     => [0.9, NaN])
        s = model_update_summary(synthetic_update_result(; metrics);
                                 input_ranges = [(0.0, 1.0), (4.0, 6.0)], n_samples = 5)
        @test occursin("Status:     success", s)
        @test occursin("Trainable parameters", s)
        @test occursin("drift explained=90", s)
        @test occursin("drift explained=n/a", s)
        @test occursin("Symbolic regression (linear basis", s)
        @test occursin("Sparse symbolic regression", s)

        # Failed, symbolic-path result without metrics or ranges.
        sym = UpdateResult(false, NaN, 0, nothing, nothing, nothing, "failed", :sym_nn, nothing, nothing)
        s2  = model_update_summary(sym)
        @test occursin("Status:     failed", s2)
        @test occursin("symbolic NN", s2)
        @test !occursin("Fit vs measurement", s2)
    end

    @testset "_characterise_nn: the UDE evidence section of the prompt" begin
        s = AIRMED._characterise_nn(synthetic_update_result();
                                    input_ranges = [(0.0, 1.0), (4.0, 6.0)], n_samples = 5)
        for section in ("NN correction sampled over 2 input(s)", "Linear regression",
                        "Sparse symbolic regression", "Partial sensitivities",
                        "Monotonicity", "Representative sample", "Temporal signature analysis")
            @test occursin(section, s)
        end
        @test occursin("did not succeed",
                       AIRMED._characterise_nn(synthetic_update_result(; success = false)))
        @test occursin("No input_ranges supplied", AIRMED._characterise_nn(synthetic_update_result()))
    end

    @testset "_auto_input_ranges" begin
        r = AIRMED._auto_input_ranges(synthetic_update_result())
        @test r[1][1] ≈ -0.05 && r[1][2] ≈ 1.05       # 5% margin on a varying input
        @test r[2][1] ≈ 4.5   && r[2][2] ≈ 5.5        # constant input: 10% of its value
        no_traj = UpdateResult(true, 0.0, 0, nothing, nothing, nothing, "")
        @test isnothing(AIRMED._auto_input_ranges(no_traj))
    end

    @testset "_mean_drift_explained" begin
        mk(m) = UpdateResult(true, 0.0, 0, nothing, nothing, nothing, "",
                             nothing, nothing, nothing, nothing, m)
        @test AIRMED._mean_drift_explained(mk(Dict{Symbol, Any}(:drift_explained => [0.2, NaN, 0.4]))) ≈ 0.3
        @test isnothing(AIRMED._mean_drift_explained(mk(Dict{Symbol, Any}())))
        @test isnothing(AIRMED._mean_drift_explained(mk(Dict{Symbol, Any}(:drift_explained => [NaN]))))
    end

    @testset "_extract_u0_vec handles each u0 form" begin
        @test AIRMED._extract_u0_vec(problem) == [0.0]                         # Pair map
        @test AIRMED._extract_u0_vec(rebuild_problem(problem; u0 = [0.5])) == [0.5]   # plain vector
        @test AIRMED._extract_u0_vec(rebuild_problem(problem; u0 = Dict(cap.v => 0.25))) == [0.25]
    end
end
