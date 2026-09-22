# Design notes

Rationale behind the parts of the pipeline whose behaviour is not evident from
the code itself. The source refers to the sections below instead of repeating
them inline.

## Hook-local characterisation

Measures the constitutive law of the missing element at each declared hook,
rather than inferring it from a global correction.

A global UDE adds one lumped term to a state derivative. It can therefore
indicate that the model is incomplete, but not where: with a single observed
dynamic state it cannot distinguish topologies with identical effective
dynamics. The hook residual can, since it is computed per hook from the sensors.

Steps, per hook:

1. Balance law from `problem.hook_residuals`, giving a (drive, response) pair
   without a model.
2. Conditioning: smooth, differentiate, integrate, and propagate the sensor
   noise.
3. SNR gate: decide whether a residual is present at all.
4. Local regression on the measured pair, supervised and without an ODE solve
   or adjoint, at a cost of about 0.5 s.
5. Sparse fit over a basis derived from `component_guesses`, with one column per
   candidate taken from its constitutive equation, so that a surviving
   coefficient is that component's parameter.

The basis in step 5 is taken from the library rather than from a generic
polynomial. Over a narrow operating range the monomials are near-collinear, so
the fit can recover a wrong decomposition with a smaller residual than the true
law. The library columns are near-orthogonal on the same data.

Implemented in `_characterise_hooks`.

### Choice of candidate inputs

The candidate inputs are every observable channel of the known components, plus
time, plus the derivative and the integral of the drive.

Only the constituents of the response are excluded. The target is their linear
combination, so retaining them would let a regressor reproduce it exactly
without identifying a law. The drive is kept: it is never one of them, since a
parallel hook is driven by a potential and responds with flows, and a series
hook the reverse. Dropping it would remove the hypothesis `response = f(drive)`,
which is the relevant one for a two-terminal element.

The derivative and the integral of the drive make energy-storing elements
expressible. A capacitor obeys `i = C*du/dt` and an inductor `u = L*di/dt`, both
relations to the rate of the drive, which a regressor cannot see from
instantaneous values alone. Storage does not appear as unequal port currents: in
a one-port the two port flows are equal and opposite even for a capacitor, since
charge is separated across the dielectric rather than accumulated at a terminal.
It appears in the dynamic relation between potential and flow. Unequal port
flows indicate a two-port, which is modelled explicitly rather than identified
here.

`terminal_inputs = true` narrows the basis to the terminal quantities of the
element itself: its drive, the derivative and integral of that drive, and time.
The constitutive law of a two-terminal component relates its own potential, flow
and, for a scheduled or cycling element, time; channels of other components do
not enter it. Including them can yield a fit in the channels of a slack element
that absorbs every imbalance. Such a fit restates conservation instead of a
constitutive law, and can reach a higher coefficient of determination than the
correct law, after which pruning removes the correct terms.

### Generic-basis symbolic regression

The sparse fit runs over the live inputs above, i.e. the measured channels of
the known components plus time. No component is assumed: the reported law is
expressed in quantities the sensors deliver, and the mapping onto a component is
left to the model. Regressing on the drive of the element and its derivative and
integral is uninformative once that drive is pinned by a stiff bus.

The basis has degree 1. Degree 2 is not identifiable at this excitation: raw
powers of a channel varying by a few percent around a large mean are nearly
collinear with the intercept, and STLSQ then returns cancelling terms. The
coefficient of determination does not reveal this, since it is similar for the
correct fit and for a spurious multi-term fit; the scaled condition number is
the diagnostic that does.

A monomial basis in the measured channels cannot express a periodic residual.
With time among the inputs the closest approximation is a ramp, so an
oscillating and a static element yield the same verdict. If the spectrum of the
residual shows a dominant line, the matching sine and cosine pair is added to
the basis, with the period measured rather than assumed.

The pair is added without being announced in the prompt. Reporting it earlier
described what was offered rather than what was found: smoothed noise carries
enough low-frequency structure for the residual of a constant load to pass the
peak gate, so the prompt stated a period for a constant element while the fit
classified it as constant. Periodicity is reported after the fit, and only if a
trigonometric term survives pruning.

### Significance of the fit

STLSQ prunes on relative coefficient size and therefore keeps a term whose
contribution is small in absolute terms but not relative to a large intercept,
which is the case when a spurious channel term and the intercept combine to
reproduce a constant. An F-test against the intercept-only null decides whether
the fit improves on the constant model; reporting the channel term without it
would assert a dependence that the data do not support.

The effective sample size is used rather than the raw count. `_hl_smooth`
averages over `smooth_w` samples, so neighbouring residuals are correlated and
only about `n/w` are independent. Using `n` inflates F by about `w` and would
declare smooth collinear ramps significant against smoothed white noise.
`_hl_accept_factor` applies the same correction.

## Existing-parameter pre-pass

### Per-channel goodness-of-fit gate

A single existing parameter that explains the drift must fit every observable,
not only the cross-channel average. The averaged drift check combines all
sensors, so a fit that reproduces all but one channel while being far off on
that one still appears clean. This occurs when a new parallel load is absorbed
by lowering an existing resistance, which matches the bus voltage and the other
branch currents but not the branch current of the component whose parameter was
changed.

Such a fit is rejected when the normalised RMS residual of the worst channel is
both non-trivial in absolute terms and an outlier relative to the median
channel, which indicates a structural fault that no single parameter can absorb
and that should proceed to the structural search. The thresholds are
`EXISTING_FIT_WORST_CHANNEL_FLOOR` and `EXISTING_FIT_WORST_CHANNEL_OUTLIER_K`.

### Exclusion of noise-floor channels

A channel whose true trajectory lies at the sensor noise floor, for example an
idle branch current that remains near zero for any value of an adaptable
parameter, carries no signal that a single-parameter fit could explain. Even an
exact fit leaves a residual equal to the noise, and after normalisation by the
RMS of that same noise-dominated channel the ratio is approximately 1 regardless
of the parameter value, since residual and scale are both noise. This artefact
would dominate the per-channel outlier check, so such channels are excluded from
it through the signal-to-noise gate. They still contribute to the averaged CUSUM
check.

The noise estimate is formed from first differences, as the mean square of
consecutive deltas divided by two. Sample-to-sample noise raises it while a
slowly varying trend or a plateau does not, so it serves as a noise estimate
without a problem-specific constant. The threshold is
`EXISTING_FIT_CHANNEL_SNR_MIN`.

### Re-anchoring for trailing plateau segments

A plateau candidate receives `data_times` and `data_states` already sliced to a
trailing segment, so the first sample lies later than `problem.tspan[1]`. If the
ODEProblem were built over the full `problem.tspan` starting from `problem.u0`,
the solver would integrate the candidate value across the skipped prefix, a
period this fit is not scored against, before reaching the segment under test.
For a state that integrates its input, such as a state of charge, that prefix
covers the earlier transient which the constant candidate value cannot explain,
and the accumulated bias enters every compared sample.

The ODEProblem therefore starts at the first sample of the segment, and each
`u0` state is re-anchored to its measured value at that sample where the state
is one of `problem.observable_states`; otherwise the entry of `problem.u0` is
used. For a full-window call this reduces to `problem.u0` and `problem.tspan`.

## Escalation on an inadequate structure

The retry loop retries while the adapted model still drifts, a channel lies
above the noise floor, or a fitted parameter reaches a plausibility bound, but
it cannot act once the budget is exhausted. In that terminal case the best
attempt may still be poor against the noise floor, and without a rejection such
a structure is returned with `success = true`, i.e. a wrong answer is reported
with the same confidence as a correct one.

`escalate_on_inadequate_structure` closes that case. It is disabled by default,
since it changes `success`, `message` and `escalation` for callers whose retries
end on a poor verdict.

The second condition covers what the 5 sigma threshold cannot detect. If
`_fit_proposal_parameters` fails before producing a loss, because the model does
not build or no fittable parameter exists, it returns with an infinite fit loss
and every proposal at its unfitted default. The worst-channel ratio then remains
zero, not because the fit is good but because it was never computed, so the
threshold reads as adequate and a proposal carrying default parameters would be
reported as successful. A fit that could not be attempted is at least as strong
an indication of an inadequate structure as one that converged to a poor value.

## Rebuilding the adapted twin

`twin_rebuild.jl` converts the diagnosis returned by `propose_model_adaptation`
into a runnable `AIRMEDProblem`, for example to re-check drift on a fresh window
or to compare base and adapted predictions against measurements.

All required inputs, i.e. the base equations, subsystems, independent variable,
component factories, initialisation guesses and solver settings, are available
on the `AIRMEDProblem` used for the adaptation. Domain-specific input is limited
to the `ComponentGuess` factories, the initial guesses, and the choice of data
source and observables.

The wiring is shared with the fitting path through `_assemble_adapted_system`,
so a structural fix is wired identically for fitting and for simulation. Series
insertions replace the existing direct connection.
