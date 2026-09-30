# Hints for names that ReactiveMP v7 removed or moved into a package of its own: an
# `UndefVarError` for one of them says what replaces it, or which package to load.

# The nodes and algorithms of the node packages, which RxInfer does not load: `using` the package
# brings them in.
const PACKAGE_OF_NAME = Dict{Symbol, String}(
    (
        name => "AutoregressiveMessagePassingRules" for
        name in (:AR, :Autoregressive, :ConjugateAR, :ARVMP, :ARsafe, :ARunsafe)
    )...,
    (
        name => "GCVMessagePassingRules" for
        name in (:GCV, :GCVApproximation, :ExponentialLinearQuadratic)
    )...,
    (name => "ProbitMessagePassingRules" for name in (:Probit, :ProbitEP))...,
    (name => "SoftDotMessagePassingRules" for name in (:SoftDot, :softdot))...,
    (
        name => "ContinuousTransitionMessagePassingRules" for
        name in (:ContinuousTransition, :CTransition, :CTVMP)
    )...,
    (
        name => "PolyaMessagePassingRules" for name in (
            :BinomialPolya,
            :BinomialPolyaApproximation,
            :MultinomialPolya,
            :MultinomialPolyaApproximation,
        )
    )...,
    (
        name => "BIFMMessagePassingRules" for
        name in (:BIFM, :BIFMHelper, :BIFMSmoother)
    )...,
    (
        name => "FlowMessagePassingRules" for name in (
            :Flow,
            :FlowApproximation,
            :FlowModel,
            :CompiledFlowModel,
            :PlanarFlow,
            :RadialFlow,
            :AdditiveCouplingLayer,
            :PermutationLayer,
            :InputLayer,
            :PermutationMatrix,
        )
    )...,
    :DiscreteTransition => "DiscreteTransitionMessagePassingRules",
    :GaussianCoupling => "GaussianCouplingMessagePassingRules",
)

# ReactiveMP v6's names that have another name, or no counterpart, in v7.
const REPLACEMENT_OF_NAME = Dict{Symbol, String}(
    Symbol("@node") => "`@define_factor_node` declares a node",
    Symbol("@rule") => "`@define_message_update_rule` defines a message rule: `@define_message_update_rule(node = N, target = :out, args = (m[:μ]::T, q[:v]::S), body = (args) -> …)`",
    Symbol("@marginalrule") => "`@define_marginal_update_rule` defines a marginal rule",
    Symbol("@average_energy") => "`@define_average_energy` defines an average energy",
    Symbol("@call_rule") => "`@call_message_update_rule(node = N, target = :out, m = (μ = …,), q = (v = …,))` calls a message rule; `getresult` of its value is the message",
    Symbol("@call_marginalrule") => "`@call_marginal_update_rule` calls a marginal rule",
    Symbol("@logscale") => "a rule declares its log scale with the `logscale` keyword of `@define_message_update_rule`",
    :Marginalisation => "a rule no longer names `Marginalisation`: its target is the `target` keyword and its algorithm the `algorithm` keyword of `@define_message_update_rule`",
    :MomentMatching => "a rule's algorithm is the `algorithm` keyword of `@define_message_update_rule`",
    :ARMeta => "`ARMeta(form, order, stype)` is the algorithm `ARVMP(form, order, stype)`, from AutoregressiveMessagePassingRules",
    :GCVMetadata => "`GCVMetadata(method)` is the algorithm `GCVApproximation(method)`, from GCVMessagePassingRules",
    :CTMeta => "the ContinuousTransition node runs under the algorithm `CTVMP(f)`, from ContinuousTransitionMessagePassingRules",
    :FlowMeta => "the Flow node runs under the algorithm `FlowApproximation(model)`, from FlowMessagePassingRules",
    :BinomialPolyaMeta => "the BinomialPolya node runs under `BinomialPolyaApproximation`, from PolyaMessagePassingRules",
    :MultinomialPolyaMeta => "the MultinomialPolya node runs under `MultinomialPolyaApproximation`, from PolyaMessagePassingRules",
    :DeltaMeta => "a Delta node's algorithm is `DeltaApproximation(method = …, inverse = …)`, or its method alone: `@algorithm begin f() -> Linearization() end`",
    :LogScaleAnnotations => "log scales are part of every message: `infer(…; logscales = true)`, then `getlogscale(q)`",
    :AddonLogScale => "log scales are part of every message: `infer(…; logscales = true)`, then `getlogscale(q)`",
    :DefaultFunctionalDependencies => "a node declares what its rules read (`@define_dependencies`), and an initial message is set with `@initialization`",
    :RequireMessageFunctionalDependencies => "a node declares what its rules read (`@define_dependencies`), and an initial message is set with `@initialization`",
    :RequireMarginalFunctionalDependencies => "a node declares what its rules read (`@define_dependencies`), and an initial marginal is set with `@initialization`",
)

"""
    RxInfer.removed_name_hint(name::Symbol) -> Union{String, Nothing}

What to use instead of `name`, a name of ReactiveMP v6 or of a node package RxInfer does not
load, or `nothing` for any other name. RxInfer adds it to an `UndefVarError` for the name.
"""
function removed_name_hint(name::Symbol)
    package = get(PACKAGE_OF_NAME, name, nothing)
    package === nothing ||
        return "`$(name)` is defined in $(package), which RxInfer does not load: `using $(package)`"
    replacement = get(REPLACEMENT_OF_NAME, name, nothing)
    replacement === nothing ||
        return "`$(name)` is from ReactiveMP v6; in v7, $(replacement). See RxInfer's migration guide from v5 to v6"
    return nothing
end

function removed_name_error_hint(io::IO, err::UndefVarError)
    hint = removed_name_hint(err.var)
    hint === nothing || print(io, "\nHint: ", hint, ".")
    return nothing
end
