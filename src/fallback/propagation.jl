export LaplacePropagation, ExpectationPropagation, NaturalGradientPropagation

import DeltaMessagePassingRules: MomentForm, NaturalGradientForm
import DeltaMessagePassingRules

"""
    RxInfer.fallback_site(scheme::FallbackScheme)

The site policy of `scheme`, the `site_policy` option it sets: cavity propagation, where each
random variable keeps a site per edge (see `ReactiveMP.compute_site`). `nothing` by default, for a
scheme that projects each product of messages as it is.
"""
fallback_site(::FallbackScheme) = nothing

"""
    RxInfer.SitePolicy(form; tol = 1e-10)

The site policy of the cavity schemes: each random variable's site on an edge is DeltaMessagePassingRules'
`site(form, message, cavity)`, the inbound message projected against its cavity, the variable's
outbound message on the edge, by `form`'s projection, a `LaplaceForm` (Laplace propagation) or a
`MomentForm` (expectation propagation). A closed message is its own site; a raw one waits for a
cavity, but a deterministic node's pushforward, which starts against a flat one, and every
message of a variable with an initial message, `@initialization`'s `μ(x)`, which is each edge's
first cavity while the other edges have no site (`ReactiveMP.seeded_cavity`); against a cavity
that is not a Gaussian a raw message passes, so that the variable's beliefs are
projected as products, as under [`LocalLaplace`](@ref). Nothing checks that a site, a
cavity or a marginal is proper: each is passed as it is computed. The next projection of an edge
starts at its last mode.

A raw message that arrives again unchanged, against the same cavity, both to `tol` relative in
natural parameters, as messages through deterministic nodes do while the variables around them
settle, takes its last site without a projection (DeltaMessagePassingRules' `same_message`). The
site is still sent: the rules downstream wait for each of their inputs anew.
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

Whether the free energy under `site_policy`, the `site_policy` option, is expectation
propagation's, `BetheFreeEnergy(cavity = true)`, in which a node without rules takes its term from
its cavities: for any site policy but [`RxInfer.NaturalGradientSitePolicy`](@ref), whose fixed
point is a stationary point of the Bethe free energy with average energies. `false` without one.
"""
cavity_free_energy(::Nothing) = false
cavity_free_energy(policy) = true

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

A [`FallbackScheme`](@ref), Laplace propagation wherever a model has no exact rules. The rule
fallback computes the missing messages as under [`LocalLaplace`](@ref), and every random
variable keeps one site per edge, the edge's inbound message times its cavity projected by
Laplace's method, divided by the cavity. The cavity is the product of the other edges' sites, the
variable's outbound message on the edge. Every belief of a variable is then a product of sites:
its marginal and its messages are one Gaussian seen from each edge, and a message without a
Gaussian of its own, such as one pulled back through a nonlinear function, is projected against
the cavity that gives it one. On a chain like stochastic volatility, the fixed point is the
global Laplace approximation, the Gaussian at the joint mode of the posterior.

Unlike [`LocalLaplace`](@ref), which projects each product of messages on its own, it
approximates each message without a closed form once, in the context of the rest of the
variable's belief, at the cost of keeping a site per edge and of a few iterations.

The free energy is expectation propagation's: a node without rules takes its term from its
cavities, every other term is Bethe's. It approximates `-log p(y)` at a fixed point but is not a
bound, so a lower value is not a better fit.

```julia
infer(model = sv(μ = μ, φ = φ, σ = σ), data = (y = y,), iterations = 10, fallback = LaplacePropagation())
```

# Keywords
- `tol`, `maxiters`: of the Newton iterations of each projection;
- `expectation`: the expectation backend, such as `ExpectationApproximations.EADefault()`;
  `nothing`, the default, for the backends loaded in the session;
- `initialization`: the initialization strategy, [`FromPriors`](@ref) by default; `nothing` for none.

A site is computed as its edge's message arrives, so each iteration is a sweep over the sites:
give `iterations`, a few for a chain.
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

A [`FallbackScheme`](@ref), expectation propagation wherever a model has no exact rules: as
[`LaplacePropagation`](@ref), with each site from the moments of its tilted density, the edge's
message times its cavity, rather than its mode. The moments come from Gauss–Hermite points placed
on the tilted density's own Laplace approximation and weighted by the density against it, so a
message much sharper than its cavity is resolved. A belief outside the sites, a variable with
one connection or whose cavity is not a Gaussian, is moment-matched in the same way, and the free
energy is expectation propagation's, which is not a bound.

```julia
infer(model = counts(φ = φ, σ² = σ²), data = (y = y,), iterations = 10, fallback = ExpectationPropagation())
```

# Keywords
- `points`: Gauss–Hermite points per dimension of each projection, `points`ᵈ in d dimensions;
- `backend`: another quadrature for the projections, any backend of ExpectationApproximations
  with a point set for a Gaussian: `SphericalRadialCubature()` takes 2d points, exact to degree
  3, where a product grid grows as `points`ᵈ; `nothing` is `GaussHermite(points)`;
- `tol`, `maxiters`: of the Newton iterations that place them;
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

The site policy of [`NaturalGradientPropagation`](@ref): each random variable's site on an edge is
DeltaMessagePassingRules' `site(form, message, cavity, previous)`, a damped natural-gradient step
of the edge's last site, `previous.η` with its velocity `previous.velocity`, both in the outcome
the engine keeps for the edge, towards the message's natural-gradient target at the variable's
marginal, in the family of the cavity: a Gaussian, a Gamma, any exponential family with a
deterministic expectation. A closed message of the cavity's family is its own site; a raw one
waits for a cavity, and a cavity in no such family lets the message pass, so that the variable's
beliefs are projected as products, by the scheme's `VariationalForm`.

Unlike [`RxInfer.SitePolicy`](@ref), it projects a message that arrives again unchanged anew:
with momentum, the same message and cavity take the site a step further. An initial message,
`@initialization`'s `μ(x)`, seeds each edge's first cavity as for the other cavity schemes,
within its family.
"""
struct NaturalGradientSitePolicy{F <: NaturalGradientForm}
    form::F
end

function ReactiveMP.compute_site(policy::NaturalGradientSitePolicy, variable, index, message, cavity, previous)
    result = DeltaMessagePassingRules.site(policy.form, message, cavity, previous)
    return (result.site, result.outcome)
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

A [`FallbackScheme`](@ref), natural-gradient propagation (Lukashchuk et al., *Information Geometry
of Message Passing*) wherever a model has no exact rules. The rule fallback sends belief
propagation's messages unprojected, as under [`LaplacePropagation`](@ref): a node's log-density,
its pushforward towards a deterministic node's output, its factor integrated against the other
edges' messages. Every random variable keeps one site per edge, in the family of the rest of its
belief, the cavity: a Gaussian, a Gamma, a Beta. A raw message's site moves at each sweep by a
damped heavy-ball step towards its natural-gradient target,

```math
\\eta^\\star = G(q)^{-1} \\operatorname{Cov}_q[T, \\ell], \\qquad
v \\leftarrow \\beta v + \\alpha (\\eta^\\star - \\eta), \\quad \\eta \\leftarrow \\eta + v,
```

with `ℓ` the message's log-density and `q` the variable's marginal, in natural parameters, the
step halved until the marginal is proper. At a fixed point each site is its target, and each
marginal a stationary point of the variational free energy over its family: under Gaussian
cavities, the fixed point of conjugate-computation variational inference, the variational Gaussian
of the joint where a model has one latent variable; under a Gamma one, the best Gamma.

A belief that has no site, a variable with one connection or whose cavity is in no family, is
projected onto its variational Gaussian, `VariationalForm`, in the unconstrained coordinates of
its domain. The free energy is Bethe's with the average energies of the nodes without rules: the
negative evidence lower bound on a Gaussian tree whose only non-conjugate factors are leaves, and
a diagnostic elsewhere, which need not decrease from one iteration to the next.

```julia
infer(model = bioassay(x = x, n = n), data = (y = y,), iterations = 50, fallback = NaturalGradientPropagation())
```

# Keywords
- `alpha`, `beta`: the step size and the momentum of every site, `0 < alpha`, `0 ≤ beta < 1`;
  `alpha = 1`, `beta = 0` jumps to the target at each sweep, which can oscillate;
- `max_step`: the longest step in natural parameters, `Inf` for none;
- `tol`: a site has settled when `‖η⋆ - η‖ ≤ tol (1 + ‖η‖)`, which the report reads;
- `expectation`: the expectation backend of the rules and of the variational Gaussians, such as
  `ExpectationApproximations.EADefault()`; `nothing`, the default, for the backends loaded in the
  session. The targets take Gauss–Hermite points for a Gaussian and Gauss–Legendre points in the
  unconstrained coordinate of any other family;
- `initialization`: the initialization strategy, [`FromPriors`](@ref) by default; `nothing` for none.

A site moves one step per sweep, so each iteration is a sweep: give `iterations`, a few tens. A raw
message's first step starts from the flat site, against its cavity, which must be proper: a prior,
or `@initialization`'s `μ(x)` for a variable with only raw messages. A deterministic node's
output whose other edge never sends a message, such as a filter's predicted state
`x_pred := f(x_prev)` before `x ~ N(x_pred, Q)`, which the filter asks nothing of, has no cavity
either: give it one with `μ(x_pred)`. Its pushforward's site then steps from the flat one towards
its target, the pushforward's moment-matched Gaussian whatever the seed, where `f` is not monotone
or `x_prev` is a vector.

In streaming inference each data update's `iterations` move the sites on from where the last
update left them, their velocities included: give each update enough sweeps for its sites to
settle, which the report reads.
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
