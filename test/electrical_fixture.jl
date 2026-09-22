"""
Electrical circuit fixture: RC circuit with unknown series/parallel resistors.

Base model (known) — built from StandardLibrary components:
    Voltage → [Resistor R1] → [Capacitor C] → Ground

True circuit (hidden, to be learned by UDE):
    Voltage → [R1 ∥ R3_unknown] → [R2_unknown] → [C] → Ground
        - R3_unknown sits in parallel with R1 (same top/bottom nodes as R1)
        - R2_unknown sits in series after R1, feeding the capacitor
    KCL gives:
        C·dV_C/dt = (V_source - V_C) / (R1_eff + R2) ,
        with R1_eff = R1 ∥ R3 = R1·R3 / (R1 + R3).

After structural_simplify the single ODE state is cap₊v(t) — the capacitor voltage.
The UDE training state u[1] maps to this variable.
Parameter order in p0 must match the destructuring in base_ode!: [r1.R, cap.C, vref.k].

Shared between tests and scripts — contains only problem definitions, no test assertions.
"""

using AIRMED
using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant
using OrdinaryDiffEq
using Lux
using ModelingToolkitNeuralNets

# ---- Circuit parameters -------------------------------------------------------

const R1_VAL     = 1000.0   # known series resistor [Ω]
const R2_UNKNOWN = 500.0    # unknown series resistor [Ω]
const R3_UNKNOWN = 2000.0   # unknown parallel resistor [Ω]
const C_VAL      = 1e-3     # capacitor [F]
const V_SRC      = 5.0      # supply voltage [V]
const T_SPAN     = (0.0, 0.05)

# ---- Base MTK model using StandardLibrary components -------------------------

@independent_variables t

@named r1      = Resistor(; R = R1_VAL)
@named cap     = Capacitor(; C = C_VAL)
@named vsource = Voltage()
@named gnd     = Ground()
@named vref    = Constant(; k = V_SRC)

eqs = [
    connect(vref.output, vsource.V),
    connect(vsource.p, r1.p),
    connect(r1.n, cap.p),
    connect(cap.n, gnd.g),
    connect(vsource.n, gnd.g),
]

@named base_circuit = ODESystem(eqs, t; systems = [r1, cap, vsource, gnd, vref])

# ---- True (hidden) circuit for synthetic data generation ----------------------
#
# True ODE: R1 ∥ R3 in series with R2 feeding the cap.
#   R1_eff = R1·R3 / (R1 + R3)
#   C·dV_C/dt = (V_src - V_C) / (R1_eff + R2)

function true_circuit!(du, u, p, t)
    V_C_val = u[1]
    r1_val, r2, r3, c, vsrc = p
    r1_eff = (r1_val * r3) / (r1_val + r3)
    du[1] = (vsrc - V_C_val) / ((r1_eff + r2) * c)
end

function generate_true_data(tspan, n_points)
    p_true = [R1_VAL, R2_UNKNOWN, R3_UNKNOWN, C_VAL, V_SRC]
    prob   = ODEProblem(true_circuit!, [0.0], tspan, p_true)
    saveat = collect(range(tspan[1], tspan[2]; length = n_points))
    sol    = solve(prob, Tsit5(); saveat, abstol = 1e-10, reltol = 1e-8)
    return sol.t, Array(sol)
end

# ---- Base ODE and NN input (for UDE training) --------------------------------
# u[1] = cap.v (capacitor voltage) after structural_simplify.
# p = [r1.R, cap.C, vref.k] — must match the pair order in p0 below.

function base_ode!(du, u, p, t)
    r1_val, c, vsrc = p
    du[1] = (vsrc - u[1]) / (r1_val * c)
end

const BASE_P = [R1_VAL, C_VAL, V_SRC]

nn_input_fn(u, t) = [u[1], oftype(u[1], V_SRC)]   # input: [cap.v, V_source]

# ---- Problem definition -------------------------------------------------------

# ---- Symbolic UDE setup (ModelingToolkitNeuralNets / tutorial approach) ------
#
# Causal MTK ODE that mirrors base_ode!:
#   dV_C/dt = (V_src - V_C) / (r1 * C) + NN([V_C, V_src])
#
# The NN is embedded symbolically using SymbolicNeuralNetwork so that
# train_symbolic_ude can train it with AutoForwardDiff() as shown in the
# ModelingToolkitNeuralNets tutorial.  (The function form is the v1.x API;
# the @SymbolicNeuralNetwork macro only exists in ModelingToolkitNeuralNets
# v2.5+, which this project cannot use while pinned to ModelingToolkit v9.)

@variables V_C_ude(t)
@parameters r1_ude = R1_VAL  C_ude = C_VAL  vsrc_ude = V_SRC

const D_ude = Differential(t)

const nn_arch_sym = Lux.Chain(
    Lux.Dense(2 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false),
)

sym_nn_elec, θ_elec = SymbolicNeuralNetwork(;
    chain     = nn_arch_sym,
    nn_name   = :sym_nn_elec,
    nn_p_name = :θ_elec,
    n_input   = 2,
    n_output  = 1,
)

sym_correction_elec(x) = sym_nn_elec(x, θ_elec)[1]

ude_eqs = [
    D_ude(V_C_ude) ~ (vsrc_ude - V_C_ude) / (r1_ude * C_ude) +
                     sym_correction_elec([V_C_ude, vsrc_ude])
]

@named symbolic_ude_circuit = ODESystem(
    ude_eqs, t;
    defaults = [V_C_ude => 0.0, r1_ude => R1_VAL, C_ude => C_VAL, vsrc_ude => V_SRC],
)
symbolic_ude_simplified = structural_simplify(symbolic_ude_circuit)

# Build the base ODEProblem — NN weights are initialised from Lux defaults stored as
# parameter defaults by @SymbolicNeuralNetwork.
const UDE_PROB = ODEProblem(symbolic_ude_simplified, [], T_SPAN)

@info "Defining AIRMED problem: Electrical Circuit with Unknown Resistors"

problem = AIRMEDProblem(;
    name  = "Electrical Circuit — Unknown Resistors",
    model = base_circuit,
    # ComponentHooks: topology positions where unknown components could be inserted.
    # port_a and port_b define the two attachment nodes.
    # Same subsystem in both → parallel; different subsystems → series.
    component_hooks = [
        ComponentHook(:r1_parallel,   (:r1, :p), (:r1, :n),
               "Unknown component in parallel with R1"),
        ComponentHook(:r1_to_cap,     (:r1, :n), (:cap, :p),
               "Unknown component in series between R1 output and capacitor input"),
    ],
    # ComponentGuesses: component types that could fill any hook position.
    # Each factory creates a named MTK component — domain-library knowledge stays
    # here in the script, never inside AIRMED's core.
    component_guesses = [
        ComponentGuess(:Resistor,  "Standard library Resistor",
            (name, val) -> Resistor(; R = val,  name = name);
            parameter_name = :R, connector_guesses = Dict(:v => 1.0, :i => 0.01)),
        ComponentGuess(:Capacitor, "Standard library Capacitor",
            (name, val) -> Capacitor(; C = val, name = name);
            parameter_name = :C, connector_guesses = Dict(:v => 1.0, :i => 0.01)),
        ComponentGuess(:Inductor,  "Standard library Inductor",
            (name, val) -> Inductor(; L = val,  name = name);
            parameter_name = :L, connector_guesses = Dict(:v => 1.0, :i => 0.01)),
    ],
    # Preamble written into any auto-generated adaptation code so it is
    # self-contained and runnable.
    model_preamble = [
        "using ModelingToolkit",
        "using ModelingToolkitStandardLibrary.Electrical",
        "using ModelingToolkitStandardLibrary.Blocks: Constant",
    ],
    optimizable_params = [:r1_R],
    data_source = FunctionDataSource(generate_true_data),
    drift_config = DriftConfig(
        method      = CUSUM(k = 0.3, h = 4.0),
        min_samples = 10,
    ),
    supervision_config = SupervisionConfig(
        use_agent             = false,
        max_auto_updates      = 5,
        human_threshold       = 0.5,
        ask_human_on_uncertainty = false,
    ),
    tspan = T_SPAN,
    u0    = [cap.v => 0.0],
    p0    = [r1.R => R1_VAL, cap.C => C_VAL, vref.k => V_SRC],
    # Only cap.v is physically measured; r1, vsource, gnd internals are not.
    # Newly proposed components (r_r1_parallel, r_r1_to_cap) also have no sensors.
    observable_states = [cap.v],
)
