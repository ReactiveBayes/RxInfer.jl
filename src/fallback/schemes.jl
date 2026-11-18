export LocalLaplace

import DeltaMessagePassingRules: ExpectationRuleFallback, VariationalForm
import Random

"""
    FallbackScheme

The abstract type of the schemes passed to `infer(...; fallback = scheme)`. A scheme defines how
a model is approximated wherever it has no exact rules, through the following methods:

- [`RxInfer.fallback_rule`](@ref): the rule fallback, which computes the missing messages,
  marginals and energies;
- [`RxInfer.fallback_form`](@ref): the default form constraint of the beliefs;
- [`RxInfer.fallback_expectation`](@ref): the expectation backend, or `nothing` for the backends
  loaded in the session;
- [`RxInfer.fallback_initialization`](@ref): an initialization strategy, or `nothing`;
- [`RxInfer.fallback_site`](@ref): a site policy for cavity propagation, or `nothing`.

Values set explicitly in `options` take precedence over the scheme: `rulefallback`,
`default_form_constraint`, `site_policy` and `context = (expectation = …,)`. Nodes with rules
still use them, so in a model where every node has rules the scheme changes only the
initialization.

The schemes are [`LocalLaplace`](@ref), [`LaplacePropagation`](@ref),
[`ExpectationPropagation`](@ref) and [`NaturalGradientPropagation`](@ref).
"""
abstract type FallbackScheme end

"""
    RxInfer.fallback_rule(scheme::FallbackScheme)

The rule fallback of `scheme`, which `infer` uses as the `rulefallback` option.
"""
function fallback_rule end

"""
    RxInfer.fallback_form(scheme::FallbackScheme)
    RxInfer.fallback_form(scheme::FallbackScheme, backend)

The form constraint of `scheme`, which `infer` uses as the `default_form_constraint` option. The
second method also receives the expectation backend of the inference, for forms that compute
expectations themselves, such as the variational Gaussians of
[`NaturalGradientPropagation`](@ref). By default it calls the first method.
"""
function fallback_form end

fallback_form(scheme::FallbackScheme, backend) = fallback_form(scheme)

"""
    RxInfer.fallback_expectation(scheme::FallbackScheme)

The expectation backend of `scheme`, or `nothing` to use the backends loaded in the session.
These are the backends registered with `BayesBase.register_expectation_backend!`, collected once
per call of `infer` into a type-stable `BayesBase.ExpectationFirstOf` that tries exact backends
first.
"""
fallback_expectation(::FallbackScheme) = nothing

"""
    RxInfer.fallback_initialization(scheme::FallbackScheme)

The [`RxInfer.InitializationStrategy`](@ref) of `scheme`, or `nothing` for none. A specification
given to `infer` is applied first.
"""
fallback_initialization(::FallbackScheme) = nothing

"""
    LocalLaplace(; tol = 1e-10, maxiters = 100, expectation = nothing, initialization = FromPriors())

A [`FallbackScheme`](@ref) that approximates each belief separately with Laplace's method
wherever a model has no exact rules. DeltaMessagePassingRules' `ExpectationRuleFallback` computes
the missing messages from the log-densities of the nodes. `LaplaceForm(; tol, maxiters)` projects
every product of messages that has no closed form, whether a marginal or a message, onto the
Gaussian at its mode, in the unconstrained coordinates of the variable's domain. A message whose
product has no mode, such as the message of a single likelihood, is left unprojected
(`nomode = :leave`), so that exact rules downstream do not receive a Gaussian that Laplace's method
could not compute; the fallback then pulls the message back through the node. The log evidence
and the free energy are computed as well.

It is the cheapest scheme, because it keeps no sites. Each product is projected separately, so a
message without a closed form is approximated once for every product it enters, each time in a
different context, and the beliefs of a variable can disagree. [`LaplacePropagation`](@ref)
projects each such message once, in the context of the rest of the variable's belief, and reaches
the global Laplace approximation on a chain. This scheme can lose track on a chain, for example in
nonlinear filtering.

```julia
infer(model = bioassay(x = x, n = n), data = (y = y,), fallback = LocalLaplace())
```

# Keywords
- `tol`, `maxiters`: the tolerance and the maximum number of Newton iterations of each
  projection;
- `expectation`: the expectation backend, such as `ExpectationApproximations.EADefault()`;
  `nothing`, the default, uses the backends loaded in the session;
- `initialization`: the initialization strategy, [`FromPriors`](@ref) by default; `nothing` for none.
"""
struct LocalLaplace{F <: LaplaceForm, E, I} <: FallbackScheme
    form::F
    expectation::E
    initialization::I
end

function LocalLaplace(;
        tol::Real = 1.0e-10, maxiters::Integer = 100, expectation = nothing,
        initialization = FromPriors(),
    )
    return LocalLaplace(LaplaceForm(; tol, maxiters), expectation, initialization)
end

fallback_rule(::LocalLaplace) = ExpectationRuleFallback()
# The marginals take the projection as it is; a message Laplace finds no mode for is left
fallback_form(scheme::LocalLaplace) =
    (q = scheme.form, μ = LaplaceForm(scheme.form.tol, scheme.form.maxiters, :leave))
fallback_expectation(scheme::LocalLaplace) = scheme.expectation
fallback_initialization(scheme::LocalLaplace) = scheme.initialization

function Base.show(io::IO, scheme::LocalLaplace)
    return print(io, "LocalLaplace(tol = ", scheme.form.tol, ", maxiters = ", scheme.form.maxiters, ")")
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
        "`fallback` takes a fallback scheme, such as `LocalLaplace()`; `ExpectationRuleFallback()` is a rule fallback, the `rulefallback` option: `options = (rulefallback = ExpectationRuleFallback(),)`",
    ),
)
check_fallback_scheme(fallback) = throw(
    ArgumentError(
        "`fallback` takes a fallback scheme, such as `LocalLaplace()`; got `$(fallback)`. A rule fallback alone is the `rulefallback` option",
    ),
)

# The options with what `scheme` sets and they do not
apply_fallback(options::ReactiveMPInferenceOptions, ::Nothing) = options

function apply_fallback(options::ReactiveMPInferenceOptions, scheme::FallbackScheme)
    with_rule = isnothing(getrulefallback(options)) ?
        setrulefallback(options, fallback_rule(scheme)) : options
    # the backend first: a form that computes expectations takes it too
    context = something(getcontext(with_rule), NamedTuple())
    given_in_options = get(context, :expectation, nothing)
    given = something(given_in_options, Some(fallback_expectation(scheme)))
    backend = isnothing(given) ?
        session_expectation_backend(get(context, :rng, Random.default_rng())) : given
    with_form = isnothing(getdefault_form_constraint(with_rule)) ?
        setdefault_form_constraint(with_rule, fallback_form(scheme, backend)) : with_rule
    with_form = isnothing(getsite_policy(with_form)) ? setsite_policy(with_form, fallback_site(scheme)) : with_form
    isnothing(given_in_options) || return with_form
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

Return `strategy` combined with the `@initialization` `specification`, which is applied first.
A scheme of `infer(...; fallback = scheme)` uses it to combine its strategy with the
`initialization` given to `infer`. Return `specification` alone when `strategy` is `nothing`, and
throw an `ArgumentError` for a strategy that does not accept a specification.
"""
with_specification(::Nothing, specification) = specification
with_specification(strategy::FromPriors, specification) =
    isnothing(specification) ? strategy : FromPriors(specification)
function with_specification(strategy::InitializationStrategy, specification)
    isnothing(specification) && return strategy
    throw(
        ArgumentError(
            "the fallback scheme initializes with `$(strategy)`, which takes no `@initialization` specification: give `initialization` a strategy instead",
        ),
    )
end
