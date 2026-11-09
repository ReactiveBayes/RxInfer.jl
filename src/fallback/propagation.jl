export LaplacePropagation, ExpectationPropagation

import DeltaMessagePassingRules: MomentForm
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
projected as products, as under [`LaplaceApproximation`](@ref). Nothing checks that a site, a
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
fallback computes the missing messages as under [`LaplaceApproximation`](@ref), and every random
variable keeps one site per edge, the edge's inbound message times its cavity projected by
Laplace's method, divided by the cavity. The cavity is the product of the other edges' sites, the
variable's outbound message on the edge. Every belief of a variable is then a product of sites:
its marginal and its messages are one Gaussian seen from each edge, and a message without a
Gaussian of its own, such as one pulled back through a nonlinear function, is projected against
the cavity that gives it one. On a chain like stochastic volatility, the fixed point is the
global Laplace approximation.

The free energy is expectation propagation's: a node without rules takes its term from its
cavities, every other term is Bethe's.

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
energy is expectation propagation's.

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
