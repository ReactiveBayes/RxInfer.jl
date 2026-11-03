export LaplaceApproximation

import DeltaMessagePassingRules: ExpectationRuleFallback
import Random

"""
    FallbackScheme

How `infer(...; fallback = scheme)` approximates a model wherever it has no exact rules: the
rule fallback that computes the missing messages, marginals and energies, the form every belief
takes, the backend of the expectations, and how the variables are initialised. A scheme gives
each through a method:

- [`RxInfer.fallback_rule`](@ref), the rule fallback;
- [`RxInfer.fallback_form`](@ref), the default form constraint;
- [`RxInfer.fallback_expectation`](@ref), the expectation backend, or `nothing` for the backends
  loaded in the session;
- [`RxInfer.fallback_initialization`](@ref), an initialization strategy, or `nothing`.

What `options` sets explicitly wins over the scheme: `rulefallback`, `default_form_constraint` and
`context = (expectation = …,)`. A node with a rule runs it, so the scheme changes nothing in a model
every node of which has rules, but its initialisation.

See [`LaplaceApproximation`](@ref).
"""
abstract type FallbackScheme end

"""
    RxInfer.fallback_rule(scheme::FallbackScheme)

The rule fallback of `scheme`, the `rulefallback` option it sets.
"""
function fallback_rule end

"""
    RxInfer.fallback_form(scheme::FallbackScheme)

The form constraint of `scheme`, the `default_form_constraint` option it sets.
"""
function fallback_form end

"""
    RxInfer.fallback_expectation(scheme::FallbackScheme)

The expectation backend of `scheme`, or `nothing` for the backends loaded in the session: those
registered with `BayesBase.register_expectation_backend!`, frozen once for each call of `infer`
into a type-stable `BayesBase.ExpectationFirstOf`, exact ones first.
"""
fallback_expectation(::FallbackScheme) = nothing

"""
    RxInfer.fallback_initialization(scheme::FallbackScheme)

The [`RxInfer.InitializationStrategy`](@ref) of `scheme`, or `nothing` for none. A specification
given to `infer` is applied first.
"""
fallback_initialization(::FallbackScheme) = nothing

"""
    LaplaceApproximation(; tol = 1e-10, maxiters = 100, expectation = nothing, initialization = FromPriors())

A [`FallbackScheme`](@ref), Laplace's approximation wherever a model has no exact rules:
DeltaMessagePassingRules' `ExpectationRuleFallback` computes the missing messages from the nodes'
log-densities, and `LaplaceForm(; tol, maxiters)` projects every belief that has no closed form
onto a Gaussian in the unconstrained coordinates of its domain. The log evidence and the free
energy are computed too.

```julia
infer(model = bioassay(x = x, n = n), data = (y = y,), fallback = LaplaceApproximation())
```

# Keywords
- `tol`, `maxiters`: of the Newton iterations of each projection;
- `expectation`: the expectation backend, such as `ExpectationApproximations.EADefault()`;
  `nothing`, the default, for the backends loaded in the session;
- `initialization`: the initialization strategy, [`FromPriors`](@ref) by default; `nothing` for none.
"""
struct LaplaceApproximation{F <: LaplaceForm, E, I} <: FallbackScheme
    form::F
    expectation::E
    initialization::I
end

function LaplaceApproximation(;
        tol::Real = 1.0e-10, maxiters::Integer = 100, expectation = nothing,
        initialization = FromPriors(),
    )
    return LaplaceApproximation(LaplaceForm(; tol, maxiters), expectation, initialization)
end

fallback_rule(::LaplaceApproximation) = ExpectationRuleFallback()
fallback_form(scheme::LaplaceApproximation) = scheme.form
fallback_expectation(scheme::LaplaceApproximation) = scheme.expectation
fallback_initialization(scheme::LaplaceApproximation) = scheme.initialization

function Base.show(io::IO, scheme::LaplaceApproximation)
    return print(io, "LaplaceApproximation(tol = ", scheme.form.tol, ", maxiters = ", scheme.form.maxiters, ")")
end

# The backends loaded in the session, frozen into a type-stable router: the one call that depends
# on them is here, before the options reach the model, so everything downstream is static.
function session_expectation_backend(rng)
    router = BayesBase.ExpectationFirstOf(BayesBase.ExpectationDynamicRouter())
    return BayesBase.with_expectation_rng(router, rng)
end

check_fallback_scheme(::Nothing) = nothing
check_fallback_scheme(::FallbackScheme) = nothing
check_fallback_scheme(fallback::ExpectationRuleFallback) = throw(
    ArgumentError(
        "`fallback` takes a fallback scheme, such as `LaplaceApproximation()`; `ExpectationRuleFallback()` is a rule fallback, the `rulefallback` option: `options = (rulefallback = ExpectationRuleFallback(),)`",
    ),
)
check_fallback_scheme(fallback) = throw(
    ArgumentError(
        "`fallback` takes a fallback scheme, such as `LaplaceApproximation()`; got `$(fallback)`. A rule fallback alone is the `rulefallback` option",
    ),
)

# The options with what `scheme` sets and they do not
apply_fallback(options::ReactiveMPInferenceOptions, ::Nothing) = options

function apply_fallback(options::ReactiveMPInferenceOptions, scheme::FallbackScheme)
    with_rule = isnothing(getrulefallback(options)) ?
        setrulefallback(options, fallback_rule(scheme)) : options
    with_form = isnothing(getdefault_form_constraint(with_rule)) ?
        setdefault_form_constraint(with_rule, fallback_form(scheme)) : with_rule
    context = something(getcontext(with_form), NamedTuple())
    isnothing(get(context, :expectation, nothing)) || return with_form
    given = fallback_expectation(scheme)
    backend = isnothing(given) ?
        session_expectation_backend(get(context, :rng, Random.default_rng())) : given
    return setcontext(with_form, merge(context, (expectation = backend,)))
end

# The initialization under `scheme`: its strategy, applying a given specification first; a given
# strategy wins
fallback_initialization(initialization, ::Nothing) = initialization
function fallback_initialization(initialization, scheme::FallbackScheme)
    initialization isa InitializationStrategy && return initialization
    return with_specification(fallback_initialization(scheme), initialization)
end

"""
    RxInfer.with_specification(strategy::InitializationStrategy, specification)

`strategy`, applying the `@initialization` `specification` first, as a scheme of `infer(...;
fallback = scheme)` combines its strategy with the `initialization` given; `specification` alone
when `strategy` is `nothing`. A strategy that takes no specification throws an `ArgumentError` when
given one.
"""
with_specification(::Nothing, specification) = specification
with_specification(strategy::FromPriors, specification) =
    isnothing(specification) ? strategy : FromPriors(specification)
function with_specification(strategy::InitializationStrategy, specification)
    isnothing(specification) && return strategy
    throw(
        ArgumentError(
            "the fallback scheme initialises with `$(strategy)`, which takes no `@initialization` specification: give `initialization` a strategy instead",
        ),
    )
end
