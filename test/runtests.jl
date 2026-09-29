using Test
using AIRMED

@testset "AIRMED" begin
    # test_electrical.jl includes electrical_fixture.jl, which defines the
    # shared `problem`, circuit components and constants (several as `const`)
    # used by every file below. Keep it first, and do not include the fixture
    # again from another file — Julia rejects redefining a `const` binding.
    include("test_electrical.jl")
    include("test_model_adaptation.jl")
    include("test_twin_rebuild.jl")
    include("test_agent_supervisor.jl")
    include("test_types.jl")
    include("test_simulation_model.jl")
    include("test_ml_update.jl")
    include("test_hook_local.jl")
    include("test_llm_backends.jl")
    include("test_run_airmed.jl")
end
