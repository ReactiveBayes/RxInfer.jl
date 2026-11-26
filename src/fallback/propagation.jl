export LaplacePropagation, ExpectationPropagation, NaturalGradientPropagation

import DeltaMessagePassingRules: MomentForm, NaturalGradientForm
import DeltaMessagePassingRules

"""
    RxInfer.fallback_site(scheme::FallbackScheme)

The site policy of `scheme`, which `infer` uses as the `site_policy` option. With a site policy,
each random variable keeps a site per edge (cavity propagation, see `ReactiveMP.compute_site`).
The default is `nothing`, for a scheme that projects each product of messages separately.
"""
fallback_site(::FallbackScheme) = nothing

"""
    RxInfer.SitePolicy(form; tol = 1e-10)

The site policy of the cavity schemes. The site of a random variable on an edge is
DeltaMessagePassingRules' `site(form, message, cavity)`: the inbound message projected against its
cavity, which is the variable's outbound message on that edge, with the projection of `form`. The
form is a `LaplaceForm` for Laplace propagation or a `MomentForm` for expectation propagation.

A closed-form message is its own site. A raw message without a cavity passes as it is, as under
belief propagation, and is projected once the other edges' sites give it one. The pushforward of
a deterministic node starts against a flat cavity, and an initial message, `μ(x)` from
`@initialization`, is the first cavity of each edge while the other edges have no site
(`ReactiveMP.seeded_cavity`). An initial marginal, as `FromPriors` gives, is not a cavity. Against a cavity that is
not Gaussian, a raw message passes through unchanged, and the variable's beliefs are projected as
products, as under [`LocalLaplace`](@ref). Sites, cavities and marginals are not checked for
properness: each is passed on as computed. The next projection on an edge starts at its last
mode.

A raw message that arrives again unchanged, against the same cavity, both to a relative `tol` in
natural parameters, reuses its last site without a new projection (DeltaMessagePassingRules'
`same_message`). This happens to messages through deterministic nodes while the variables around
them settle. The site is still sent, because the rules downstream wait for each of their inputs
again.
"""
struct SitePolicy{F, T <: Real}
    form::F
    tol::T
end

SitePolicy(form; tol::Real = 1.0e-10) = SitePolicy(form, tol)

function ReactiveMP.compute_site(policy::SitePolicy, variable, index, message, cavity, previous)
    if previous isa NamedTuple && haskey(previous, :message) &&
            DeltaMessagePassingRules.same_density(previous.cavity, cavity, policy.tol) &&
            DeltaMessagePassingRules.same_message(previous.message, message, policy.tol)
        return (previous.site, previous)
    end
    mode = previous isa NamedTuple ? get(previous, :mode, nothing) : nothing
    result = DeltaMessagePassingRules.site(policy.form, message, cavity, mode)
    return (result.site, merge(result.outcome, (; message, cavity, result.site)))
end

"""
    RxInfer.cavity_free_energy(site_policy) -> Bool

Return whether the free energy under `site_policy` (the `site_policy` option) is that of
expectation propagation, `BetheFreeEnergy(cavity = true)`, in which a node without rules takes its
term from its cavities. This is `true` for every site policy except
[`RxInfer.NaturalGradientSitePolicy`](@ref), whose fixed point is a stationary point of the Bethe
free energy with average energies, and `false` without a site policy.
"""
cavity_free_energy(::Nothing) = false
cavity_free_energy(policy) = true

# Whether the free energy of the inference is expectation propagation's. Its terms of the nodes
# without rules come from the rule fallback, whose energy must be the one the site policy's free
# energy takes: a rule fallback given in the options, which take precedence over a scheme, may
# compute the other one, and the free energy would then be silently wrong.
function free_energy_mode(options)
    policy = getsite_policy(options)
    cavity = cavity_free_energy(policy)
    check_fallback_energy(getrulefallback(options), policy, cavity)
    return cavity
end

check_fallback_energy(fallback, policy, cavity) = nothing
function check_fallback_energy(fallback::DeltaMessagePassingRules.ExpectationRuleFallback, policy, cavity)
    isnothing(policy) && return nothing
    expected = cavity ? :cavity : :average
    fallback.energy === expected && return nothing
    throw(
        ArgumentError(
            "the free energy under the site policy `$(policy)` takes the terms of the nodes without rules from the rule fallback's `:$(expected)` energy, but the `rulefallback` given in `options` computes `:$(fallback.energy)`. Pass `ExpectationRuleFallback(form; energy = :$(expected))`, or leave `rulefallback` to the scheme",
        ),
    )
end

# An initial message, `@initialization`'s `μ(x)`, as every edge's first cavity: its natural
# parameters, shared among the other edges, for each that has no site yet
function ReactiveMP.seeded_cavity(::SitePolicy, cavity, initial, waiting, degree)
    initial isa DeltaMessagePassingRules.GaussianMessage || return cavity
    (ismissing(cavity) || cavity isa DeltaMessagePassingRules.GaussianMessage) || return cavity
    ξ0, W0 = weightedmean_precision(initial)
    share = waiting / (degree - 1)
    ξ, W = ismissing(cavity) ? (zero(ξ0), zero(W0)) : weightedmean_precision(cavity)
    return DeltaMessagePassingRules.weighted_mean_precision(ξ + share * ξ0, W + share * W0)
end

"""
    LaplacePropagation(; tol = 1e-10, maxiters = 100, expectation = nothing, initialization = FromPriors())

A [`FallbackScheme`](@ref) that runs Laplace propagation wherever a model has no exact rules.
The rule fallback computes the missing messages as under [`LocalLaplace`](@ref). Every random
variable keeps one site per edge: the edge's inbound message times its cavity, projected by
Laplace's method, divided by the cavity. The cavity of an edge is the product of the sites of the
other edges, which is the variable's outbound message on that edge. Every belief of a variable is
then a product of the same sites, so its marginal and its messages agree. A message that has no
Gaussian approximation by itself, such as one pulled back through a nonlinear function, is
projected as its product with the cavity, which has one. On a chain such as a stochastic
volatility model, the fixed point is the global Laplace approximation, the Gaussian at the joint
mode of the posterior.

Compared with [`LocalLaplace`](@ref), which projects each product of messages separately, it
approximates each message without a closed form once, in the context of the rest of the
variable's belief. The cost is a site per edge and a few iterations.

The free energy is that of expectation propagation: a node without rules takes its term from its
cavities, and all other terms are Bethe terms. At a fixed point it approximates `-log p(y)`, but
it is not a bound, so a lower value does not mean a better fit.

```julia
infer(model = sv(μ = μ, φ = φ, σ = σ), data = (y = y,), iterations = 10, fallback = LaplacePropagation())
```

# Keywords
- `tol`, `maxiters`: the tolerance and the maximum number of Newton iterations of each
  projection;
- `expectation`: the expectation backend, such as `ExpectationApproximations.EADefault()`;
  `nothing`, the default, uses the backends loaded in the session;
- `initialization`: the initialization strategy, [`FromPriors`](@ref) by default; `nothing` for none.

A site is computed when the message on its edge arrives, so each iteration is one sweep over the
sites. Set `iterations`; a few are enough for a chain.
"""
struct LaplacePropagation{F <: LaplaceForm, E, I} <: FallbackScheme
    form::F
    expectation::E
    initialization::I
end

function LaplacePropagation(; tol::Real = 1.0e-10, maxiters::Integer = 100, expectation = nothing, initialization = FromPriors())
    return LaplacePropagation(LaplaceForm(; tol, maxiters), expectation, initialization)
end

fallback_rule(scheme::LaplacePropagation) = ExpectationRuleFallback(scheme.form)
fallback_form(scheme::LaplacePropagation) =
    (q = scheme.form, μ = LaplaceForm(scheme.form.tol, scheme.form.maxiters, :leave))
fallback_site(scheme::LaplacePropagation) = SitePolicy(scheme.form)
fallback_expectation(scheme::LaplacePropagation) = scheme.expectation
fallback_initialization(scheme::LaplacePropagation) = scheme.initialization

Base.show(io::IO, scheme::LaplacePropagation) =
    print(io, "LaplacePropagation(tol = ", scheme.form.tol, ", maxiters = ", scheme.form.maxiters, ")")

"""
    ExpectationPropagation(; points = 20, backend = nothing, tol = 1e-10, maxiters = 100, expectation = nothing, initialization = FromPriors())

A [`FallbackScheme`](@ref) that runs expectation propagation wherever a model has no exact
rules. It works as [`LaplacePropagation`](@ref), but each site comes from the moments of its
tilted density, the edge's message times its cavity, instead of from its mode. The moments are
computed with Gauss–Hermite points placed on the Laplace approximation of the tilted density and
weighted by the ratio of the density to that approximation, so a message that is much sharper than
its cavity is resolved. A belief without sites, of a variable with one connection or whose cavity
is not Gaussian, is moment-matched in the same way. The free energy is that of expectation
propagation, which is not a bound.

```julia
infer(model = counts(φ = φ, σ² = σ²), data = (y = y,), iterations = 10, fallback = ExpectationPropagation())
```

# Keywords
- `points`: the number of Gauss–Hermite points per dimension of each projection, `points`ᵈ in
  d dimensions;
- `backend`: a different quadrature for the projections, any backend of ExpectationApproximations
  with a point set for a Gaussian. For example, `SphericalRadialCubature()` uses 2d points and is
  exact to degree 3, where a product grid grows as `points`ᵈ. `nothing` means
  `GaussHermite(points)`;
- `tol`, `maxiters`: the tolerance and the maximum number of Newton iterations that place the
  points;
- `expectation`, `initialization`: as for [`LaplacePropagation`](@ref).
"""
struct ExpectationPropagation{F <: MomentForm, E, I} <: FallbackScheme
    form::F
    expectation::E
    initialization::I
end

function ExpectationPropagation(; points::Integer = 20, backend = nothing, tol::Real = 1.0e-10, maxiters::Integer = 100, expectation = nothing, initialization = FromPriors())
    return ExpectationPropagation(MomentForm(; tol, maxiters, points, backend), expectation, initialization)
end

fallback_rule(scheme::ExpectationPropagation) = ExpectationRuleFallback(scheme.form)
fallback_form(scheme::ExpectationPropagation) =
    (q = scheme.form, μ = MomentForm(scheme.form.tol, scheme.form.maxiters, :leave, scheme.form.points, scheme.form.backend))
fallback_site(scheme::ExpectationPropagation) = SitePolicy(scheme.form)
fallback_expectation(scheme::ExpectationPropagation) = scheme.expectation
fallback_initialization(scheme::ExpectationPropagation) = scheme.initialization

Base.show(io::IO, scheme::ExpectationPropagation) =
    print(io, "ExpectationPropagation(backend = ", scheme.form.backend, ", tol = ", scheme.form.tol, ", maxiters = ", scheme.form.maxiters, ")")

"""
    RxInfer.NaturalGradientSitePolicy(form::NaturalGradientForm)

The site policy of [`NaturalGradientPropagation`](@ref). The site of a random variable on an edge
is DeltaMessagePassingRules' `site(form, message, cavity, previous)`: a damped natural-gradient
step from the edge's last site towards the natural-gradient target of the message at the
variable's marginal. The step stays in the family of the cavity, which is a Gaussian, a Gamma or
another exponential family with a deterministic expectation. The last site, `previous.η`, and its
velocity, `previous.velocity`, are part of the outcome that the engine keeps for the edge.

A closed-form message in the family of the cavity is its own site. A raw message waits for a
cavity. Against a cavity that is in no such family, the message passes through unchanged, and the
variable's beliefs are projected as products by the scheme's `VariationalForm`.

Unlike [`RxInfer.SitePolicy`](@ref), it updates the site again when a message arrives unchanged:
with momentum, the same message and cavity move the site a step further. An initial message,
`μ(x)` from `@initialization`, gives the first cavity of each edge as for the other cavity
schemes, within its family.
"""
struct NaturalGradientSitePolicy{F <: NaturalGradientForm}
    form::F
end

function ReactiveMP.compute_site(policy::NaturalGradientSitePolicy, variable, index, message, cavity, previous)
    result = DeltaMessagePassingRules.site(policy.form, message, cavity, previous)
    return (result.site, result.outcome)
end

# A closed message is its own site, in whatever representation the form keeps it: a Gamma as a
# `GammaShapeRate`, or a cached site equal to the message. A raw message that passes while its
# cavity has not formed (`:waiting`) is not: the engine projects it again, as it does a site.
function ReactiveMP.is_own_site(::Union{SitePolicy, NaturalGradientSitePolicy}, message, site, outcome)
    method = outcome isa NamedTuple ? get(outcome, :method, nothing) : nothing
    method === :waiting && return false
    return site === message || method === :closed_form
end

# The initial message's natural parameters, shared among the other edges, added to the cavity's
# where both are of one family
function ReactiveMP.seeded_cavity(::NaturalGradientSitePolicy, cavity, initial, waiting, degree)
    seed = DeltaMessagePassingRules.family_parameters(initial)
    isnothing(seed) && return cavity
    family, η0, conditioner = seed
    share = waiting / (degree - 1)
    ismissing(cavity) && return DeltaMessagePassingRules.family_site(family, share .* η0, conditioner)
    current = DeltaMessagePassingRules.family_parameters(cavity)
    (isnothing(current) || current[1] !== family) && return cavity
    return DeltaMessagePassingRules.family_site(family, current[2] .+ share .* η0, conditioner)
end

# Bethe's free energy with average energies, the objective whose stationary point is the fixed point
cavity_free_energy(::NaturalGradientSitePolicy) = false

"""
    NaturalGradientPropagation(; alpha = 0.5, beta = 0.2, max_step = Inf, tol = 1e-8, expectation = nothing, initialization = FromPriors())

A [`FallbackScheme`](@ref) that runs natural-gradient propagation (Lukashchuk et al.,
*Information Geometry of Message Passing*) wherever a model has no exact rules. As under
[`LaplacePropagation`](@ref), the rule fallback sends the belief propagation messages unprojected:
the log-density of a node, the pushforward of a deterministic node towards its output, or the
factor integrated against the messages of the other edges. Every random variable keeps one site
per edge, in the family of its cavity, the rest of its belief: a Gaussian, a Gamma or a Beta. At
each sweep, the site of a raw message moves by a damped heavy-ball step towards its
natural-gradient target,

```math
\\eta^\\star = G(q)^{-1} \\operatorname{Cov}_q[T, \\ell], \\qquad
v \\leftarrow \\beta v + \\alpha (\\eta^\\star - \\eta), \\quad \\eta \\leftarrow \\eta + v,
```

where `ℓ` is the log-density of the message and `q` is the marginal of the variable, in natural
parameters. The step is halved until the marginal is proper. At a fixed point, each site equals
its target and each marginal is a stationary point of the variational free energy over its
family. With Gaussian cavities, this is the fixed point of conjugate-computation variational
inference, which for a model with one latent variable is the variational Gaussian of the joint;
with a Gamma cavity, it is the best Gamma.

A belief without sites, of a variable with one connection or whose cavity is in no family, is
projected onto its variational Gaussian, `VariationalForm`, in the unconstrained coordinates of its
domain. The free energy is the Bethe free energy with the average energies of the nodes without
rules. On a Gaussian tree whose only non-conjugate factors are leaves, it is the negative evidence
lower bound; elsewhere it is a diagnostic and need not decrease from one iteration to the next.

```julia
infer(model = bioassay(x = x, n = n), data = (y = y,), iterations = 50, fallback = NaturalGradientPropagation())
```

# Keywords
- `alpha`, `beta`: the step size and the momentum of every site, with `0 < alpha` and
  `0 ≤ beta < 1`; `alpha = 1`, `beta = 0` jumps to the target at each sweep, which can oscillate;
- `max_step`: the largest step in natural parameters, `Inf` for no limit;
- `tol`: a site has settled when `‖η⋆ - η‖ ≤ tol (1 + ‖η‖)`; the report uses this test;
- `expectation`: the expectation backend of the rules and of the variational Gaussians, such as
  `ExpectationApproximations.EADefault()`; `nothing`, the default, uses the backends loaded in the
  session. The targets use Gauss–Hermite points for a Gaussian and Gauss–Legendre points in the
  unconstrained coordinate of any other family, or a closed form where ClosedFormExpectations has
  one, once it is loaded (`using ClosedFormExpectations`);
- `initialization`: the initialization strategy, [`FromPriors`](@ref) by default; `nothing` for none.

A site moves one step per sweep, and each iteration is one sweep, so set `iterations` to a few
tens. The first step for a raw message starts from a flat site and needs a proper cavity: a prior,
or `μ(x)` from `@initialization` for a variable with only raw messages. The output of a
deterministic node whose other edge never sends a message has no cavity either. An example is the
predicted state of a filter, `x_pred := f(x_prev)` before `x ~ N(x_pred, Q)`, from which the filter
requests nothing. Give it a cavity with `μ(x_pred)`. Where `f` is not monotone or `x_prev` is a
vector, the site of the pushforward then steps from the flat site towards its target, the
moment-matched Gaussian of the pushforward, whatever the initial message.

In streaming inference, the `iterations` of each data update continue from the sites, including
their velocities, that the previous update left. Give each update enough sweeps for its sites to
settle; the report shows whether they did.
"""
struct NaturalGradientPropagation{F <: NaturalGradientForm, E, I} <: FallbackScheme
    form::F
    expectation::E
    initialization::I
end

function NaturalGradientPropagation(; alpha::Real = 0.5, beta::Real = 0.2, max_step::Real = Inf, tol::Real = 1.0e-8, expectation = nothing, initialization = FromPriors())
    return NaturalGradientPropagation(NaturalGradientForm(; alpha, beta, max_step, tol), expectation, initialization)
end

fallback_rule(scheme::NaturalGradientPropagation) = ExpectationRuleFallback(scheme.form)
# The beliefs without sites through the inference's backend; a message with no mode is left
fallback_form(scheme::NaturalGradientPropagation, backend) =
    (q = VariationalForm(1.0e-10, 100, :project, backend), μ = VariationalForm(1.0e-10, 100, :leave, backend))
fallback_form(scheme::NaturalGradientPropagation) = fallback_form(scheme, nothing)
fallback_site(scheme::NaturalGradientPropagation) = NaturalGradientSitePolicy(scheme.form)
fallback_expectation(scheme::NaturalGradientPropagation) = scheme.expectation
fallback_initialization(scheme::NaturalGradientPropagation) = scheme.initialization

# The variational Gaussians take their iterations from a form given in the options
maxiters_hint(::NaturalGradientPropagation) = "a `VariationalForm(maxiters = …)` given as the option `default_form_constraint`"

Base.show(io::IO, scheme::NaturalGradientPropagation) = print(
    io, "NaturalGradientPropagation(alpha = ", scheme.form.alpha, ", beta = ", scheme.form.beta,
    ", max_step = ", scheme.form.max_step, ", tol = ", scheme.form.tol, ")",
)
