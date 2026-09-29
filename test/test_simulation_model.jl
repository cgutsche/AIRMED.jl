"""
Simulation and model-utility tests: `CSVDataSource` round-trip, time-point
interpolation, plain-ODE simulation and the small ModelingToolkit helpers in
model.jl.

Relies on `problem`, `base_ode!`, `nn_input_fn`, `BASE_P`, `T_SPAN` and the
electrical-circuit constants already defined by electrical_fixture.jl, which
test_electrical.jl includes once — see the ordering note in runtests.jl.
"""

using Test
using CSV
using DataFrames
using Random
using ModelingToolkit

@testset "Simulation & Model Utilities" begin

    @testset "CSVDataSource round-trips through load_data" begin
        mktempdir() do dir
            path = joinpath(dir, "data.csv")
            df = DataFrame(t = [0.0, 1.0, 2.0, 3.0, 4.0], v = [0.0, 1.0, 4.0, 9.0, 16.0])
            CSV.write(path, df)

            src = CSVDataSource(path, :t, [:v])
            times, states = load_data(src, (0.0, 3.0))

            @test times == [0.0, 1.0, 2.0, 3.0]
            @test size(states) == (1, 4)
            @test vec(states) == [0.0, 1.0, 4.0, 9.0]
        end
    end

    @testset "FunctionDataSource dispatches to the generator" begin
        data_times, data_states = load_data(problem.data_source, T_SPAN, 50)
        @test length(data_times) == 50
        @test size(data_states, 2) == 50
    end

    @testset "interpolate_to_times matches the underlying solution" begin
        sol    = simulate(problem)
        target = collect(range(T_SPAN...; length = 10))
        interp = interpolate_to_times(sol, target)
        @test size(interp) == (1, 10)
        @test all(isfinite, interp)
        @test isapprox(interp[1, 1], sol(target[1])[1]; atol = 1e-9)
    end

    @testset "simulate_ude runs a plain ODE function" begin
        sol = simulate_ude(base_ode!, [0.0], T_SPAN, BASE_P)
        @test sol.retcode == ReturnCode.Success
        @test all(isfinite, Array(sol))
    end

    @testset "make_ode_problem builds a solvable ODEProblem" begin
        prob = make_ode_problem(problem)
        sol  = solve(prob, Tsit5())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "make_plain_ode_problem honours saveat" begin
        prob = AIRMED.make_plain_ode_problem(base_ode!, [0.0], T_SPAN, BASE_P; saveat = 0.01)
        @test length(solve(prob, Tsit5()).t) == 6
        @test isempty(AIRMED.make_plain_ode_problem(base_ode!, [0.0], T_SPAN, BASE_P).kwargs)
    end

    @testset "compute_residuals interpolates onto the measurement times" begin
        sim_t = [0.0, 1.0, 2.0]
        sim_x = [0.0 10.0 20.0]
        _, r = compute_residuals(sim_t, sim_x, [0.5, 1.5, 2.0], [5.0 16.0 20.0])
        @test r ≈ [0.0 1.0 0.0]
    end

    @testset "detect_drift on a residual matrix averages across states" begin
        shift = vcat(zeros(30), fill(1.0, 30))
        R     = vcat(permutedims(shift), permutedims(-shift))   # signs cancel unless |.| is taken
        cfg   = DriftConfig(method = SimpleThreshold(threshold = 0.5), min_samples = 10)
        @test detect_drift(R, cfg).drift_detected
        @test AIRMED._method_symbol(CUSUM()) == :cusum
        @test AIRMED._method_symbol(EWMA()) == :ewma
        @test AIRMED._method_symbol(SimpleThreshold()) == :threshold
    end

    @testset "get_state_names / get_param_names / model_summary" begin
        names_s = get_state_names(problem.simplified_model)
        names_p = get_param_names(problem.simplified_model)
        @test length(names_s) == length(ModelingToolkit.unknowns(problem.simplified_model))
        @test length(names_p) == length(ModelingToolkit.parameters(problem.simplified_model))
        @test !isempty(names_s) && all(n -> n isa Symbol, names_s)
        @test !isempty(names_p) && all(n -> n isa Symbol, names_p)
        # model_summary just prints a description; it must run without error.
        @test model_summary(problem.simplified_model) === nothing
    end

    @testset "add_nn_to_ode wraps the base ODE with a network correction" begin
        nn = build_nn(2, 4, 1; depth = 1)
        ps, st = Lux.setup(Random.default_rng(), nn)
        state_ref = Ref(st)
        ude! = add_nn_to_ode(base_ode!, nn, state_ref, nn_input_fn)

        du = [0.0]
        ude!(du, [0.0], (; base = BASE_P, nn = ps), 0.0)
        @test isfinite(du[1])
    end
end
