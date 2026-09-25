import GraphPPL
import MacroTools
import ExponentialFamily
import Static

import MacroTools: @capture

"""
A backend for GraphPPL that uses ReactiveMP for inference.
"""
struct ReactiveMPGraphPPLBackend{T}
    should_contract_node::T
end

# Model specification with `@model` macro

function GraphPPL.model_macro_interior_pipelines(::ReactiveMPGraphPPLBackend)
    default_pipelines = GraphPPL.model_macro_interior_pipelines(
        GraphPPL.DefaultBackend()
    )
    return (
        RxInfer.error_datavar_constvar_randomvar,
        RxInfer.compose_simple_operators_with_brackets,
        RxInfer.inject_tilderhs_aliases,
        default_pipelines...,
    )
end

"""
    error_datavar_constvar_randomvar(expr::Expr)

An additional pipeline stage for the `@model` macro from `GraphPPL`.
Notifies the user that the `datavar`, `constvar` and `randomvar` syntax has been removed and is no longer supported in the current version.
"""
function error_datavar_constvar_randomvar(e::Expr)
    if @capture(
        e,
        (
            (lhs_ = datavar(args__)) | (lhs_ = constvar(args__)) |
            (lhs_ = randomvar(args__))
        )
    )
        return :(error(
            "`datavar`, `constvar` and `randomvar` syntax has been removed from new versions of `RxInfer.jl`. Please refer to `GraphPPL` documentation for new model creation syntax.",
        ))
    end
    return e
end

"""
    compose_simple_operators_with_brackets(expr::Expr)

An additional pipeline stage for the `@model` macro from `GraphPPL`. 
This pipeline converts simple multi-argument operators to their corresponding bracketed expression. 
E.g. the expression `x ~ x1 + x2 + x3 + x4` becomes `x ~ ((x1 + x2) + x3) + x4`.
The operators to compose are `+` and `*`.
"""
function compose_simple_operators_with_brackets(e::Expr)
    operators_to_compose = (:+, :*)
    if @capture(e, lhs_ ~ rhs_)
        newrhs = MacroTools.postwalk(rhs) do subexpr
            for operator in operators_to_compose
                if @capture(subexpr, $(operator)(args__))
                    return recursive_brackets_expression(operator, args)
                end
            end
            return subexpr
        end
        return :($lhs ~ $newrhs)
    end
    return e
end

function recursive_brackets_expression(operator, args)
    if length(args) > 2
        return recursive_brackets_expression(
            operator,
            vcat([Expr(:call, operator, args[1], args[2])], args[3:end]),
        )
    else
        return Expr(:call, operator, args...)
    end
end

function show_tilderhs_alias(io = stdout)
    foreach(skipmissing(map(last, ReactiveMPNodeAliases))) do alias
        println(io, "- ", alias)
    end
end

function apply_alias_transformation(notanexpression, alias)
    # We always short-circuit on non-expression
    return (notanexpression, true)
end

function apply_alias_transformation(expression::Expr, alias)
    _expression = first(alias)(expression)
    # Returns potentially modified expression and a Boolean flag, 
    # which indicates if expression actually has been modified
    return (_expression, _expression !== expression)
end

"""
    inject_tilderhs_aliases(e::Expr)

A pipeline stage for the `@model` macro from `GraphPPL`.
This pipeline applies the aliases defined in `ReactiveMPNodeAliases` to the expression.
"""
function inject_tilderhs_aliases(e::Expr)
    if @capture(e, lhs_ ~ rhs_)
        newrhs = MacroTools.postwalk(rhs) do expression
            # We short-circuit if `mflag` is true
            _expression, _ = foldl(
                ReactiveMPNodeAliases; init = (expression, false)
            ) do (expression, mflag), alias
                return if mflag
                    (expression, true)
                else
                    apply_alias_transformation(expression, alias)
                end
            end
            return _expression
        end
        return :($lhs ~ $newrhs)
    else
        return e
    end
end

"""
Syntactic sugar for `ReactiveMP` nodes.
Replaces `a || b` with `StandardMessagePassingRules.OR(a, b)`, `a && b` with `StandardMessagePassingRules.AND(a, b)`, `a -> b` with `StandardMessagePassingRules.IMPLY(a, b)` and `¬a` with `StandardMessagePassingRules.NOT(a)`.
"""
const ReactiveMPNodeAliases = (
    (
        (expression) -> if @capture(expression, a_ || b_)
            :(StandardMessagePassingRules.OR($a, $b))
        else
            expression
        end,
        "`a || b`: alias for `StandardMessagePassingRules.OR(a, b)` node (operator precedence between `||`, `&&`, `->` and `!` is the same as in Julia).",
    ),
    (
        (expression) -> if @capture(expression, a_ && b_)
            :(StandardMessagePassingRules.AND($a, $b))
        else
            expression
        end,
        "`a && b`: alias for `StandardMessagePassingRules.AND(a, b)` node (operator precedence `||`, `&&`, `->` and `!` is the same as in Julia).",
    ),
    (
        (expression) -> if @capture(expression, a_ -> b_)
            :(StandardMessagePassingRules.IMPLY($a, $b))
        else
            expression
        end,
        "`a -> b`: alias for `StandardMessagePassingRules.IMPLY(a, b)` node (operator precedence `||`, `&&`, `->` and `!` is the same as in Julia).",
    ),
    (
        (expression) -> if @capture(expression, (¬a_) | (!a_))
            :(StandardMessagePassingRules.NOT($a))
        else
            expression
        end,
        "`¬a` and `!a`: alias for `StandardMessagePassingRules.NOT(a)` node (Unicode `\\neg`, operator precedence `||`, `&&`, `->` and `!` is the same as in Julia).",
    ),
)

export @model

# This is a special `@model` macro that uses `ReactiveMP` backend

"""
```julia
@model function model_name(model_arguments...)
    # model description
end
```

`@model` macro generates a function that returns an equivalent graph-representation of the given probabilistic model description.
See the documentation to [`GraphPPL.@model`](https://github.com/ReactiveBayes/GraphPPL.jl) for more information.

## Supported aliases in the model specification specifically for RxInfer.jl and ReactiveMP.jl
$(sprint(RxInfer.show_tilderhs_alias))
"""
macro model(model_specification)
    return esc(
        GraphPPL.model_macro_interior(
            ReactiveMPGraphPPLBackend{Static.False}, model_specification
        ),
    )
end

# Backend specific methods

# A node is what a rule package declares with `@define_factor_node`: its declaration answers
# GraphPPL's questions. A function that is not a declared node is a Delta node, as in v6.
isdeclarednode(something) = applicable(MessagePassingRulesBase.nodespec, something)

function GraphPPL.NodeBehaviour(
        backend::ReactiveMPGraphPPLBackend, something::F
    ) where {F}
    isdeclarednode(something) || return undeclared_node_behaviour(something)
    return node_behaviour(MessagePassingRulesBase.sdtype(something))
end
node_behaviour(::MessagePassingRulesBase.Deterministic) = GraphPPL.Deterministic()
node_behaviour(::MessagePassingRulesBase.Stochastic) = GraphPPL.Stochastic()
# v6's defaults for what no package declares: a function or a type is deterministic, a
# distribution stochastic.
undeclared_node_behaviour(::Type{<:Distribution}) = GraphPPL.Stochastic()
undeclared_node_behaviour(::Distribution) = GraphPPL.Stochastic()
undeclared_node_behaviour(::Union{Function, Type}) = GraphPPL.Deterministic()
undeclared_node_behaviour(something) = error(
    "`$(something)` is not a factor node: no loaded package declares it with `@define_factor_node`, and it is not a function, which would be a Delta node",
)

# If node contraction is enabled, a declared node is atomic; anything else falls back to the
# `DefaultBackend`.
function GraphPPL.NodeType(
        backend::ReactiveMPGraphPPLBackend{Static.True}, something::F
    ) where {F}
    isdeclarednode(something) && return GraphPPL.Atomic()
    return GraphPPL.NodeType(ReactiveMPGraphPPLBackend(Static.False()), something)
end

# Fallback to the default behaviour
function GraphPPL.NodeType(
        ::ReactiveMPGraphPPLBackend{Static.False}, something::F
    ) where {F}
    return GraphPPL.NodeType(GraphPPL.DefaultBackend(), something)
end
function GraphPPL.aliases(
        ::ReactiveMPGraphPPLBackend{Static.False}, something::F
    ) where {F}
    # Fallback to the default behaviour
    return GraphPPL.aliases(GraphPPL.DefaultBackend(), something)
end

# A declared node's interfaces, a group once by its name; a node with a group may be given any
# number of its members, which `factornode` checks.
function GraphPPL.interfaces(
        backend::ReactiveMPGraphPPLBackend, something::F, ninputs
    ) where {F}
    isdeclarednode(something) || return GraphPPL.interfaces(GraphPPL.DefaultBackend(), something, ninputs)
    names = MessagePassingRulesBase.interfaces(something)
    groups = MessagePassingRulesBase.interface_groups(something)
    # A trailing group given no members, as DiscreteTransition's `T` with none, is left out.
    if isequal(length(names), ninputs + 1) && last(names) in groups
        return GraphPPL.StaticInterfaces(Base.front(names))
    end
    if isequal(length(names), ninputs) || !isempty(groups)
        return GraphPPL.StaticInterfaces(names)
    end
    return error("`$(something)` has `$(length(names))` interfaces `$(names)`, but `$(ninputs)` requested.")
end

function GraphPPL.factor_alias(
        ::ReactiveMPGraphPPLBackend, something::F, interfaces
    ) where {F}
    # Fallback to the default behaviour
    return GraphPPL.factor_alias(
        GraphPPL.DefaultBackend(), something, interfaces
    )
end
function GraphPPL.interface_aliases(
        ::ReactiveMPGraphPPLBackend, something::F
    ) where {F}
    # Fallback to the default behaviour
    return GraphPPL.interface_aliases(GraphPPL.DefaultBackend(), something)
end

# The positional arguments of a declared atomic node are its interfaces after `out`, in order;
# a trailing group takes the arguments left, as DiscreteTransition's `T` does.
function GraphPPL.default_parametrization(
        backend::ReactiveMPGraphPPLBackend, nodetype, something::F, rhs
    ) where {F}
    if nodetype isa GraphPPL.Atomic && isdeclarednode(something)
        inputs = Base.tail(MessagePassingRulesBase.interfaces(something))
        isequal(length(inputs), length(rhs)) && return NamedTuple{inputs}(rhs)
        groups = MessagePassingRulesBase.interface_groups(something)
        if !isempty(inputs) && last(inputs) in groups && length(rhs) >= length(inputs) - 1
            fixed = length(inputs) - 1
            members = rhs[(fixed + 1):end]
            return isempty(members) ? NamedTuple{inputs[1:fixed]}(rhs[1:fixed]) :
                NamedTuple{inputs}((rhs[1:fixed]..., collect(members)))
        end
        return error("`$(something)` has `$(length(inputs))` input interfaces `$(inputs)`, but `$(length(rhs))` arguments provided.")
    end
    return GraphPPL.default_parametrization(GraphPPL.DefaultBackend(), nodetype, something, rhs)
end

function GraphPPL.instantiate(::Type{ReactiveMPGraphPPLBackend})
    return ReactiveMPGraphPPLBackend(Static.False())
end

function GraphPPL.instantiate(::Type{ReactiveMPGraphPPLBackend{Static.True}})
    return ReactiveMPGraphPPLBackend(Static.True())
end
function GraphPPL.instantiate(::Type{ReactiveMPGraphPPLBackend{Static.False}})
    return ReactiveMPGraphPPLBackend(Static.False())
end

# Node specific aliases

GraphPPL.factor_alias(
    ::ReactiveMPGraphPPLBackend,
    ::Type{Normal},
    ::GraphPPL.StaticInterfaces{(:μ, :v)},
) = ExponentialFamily.NormalMeanVariance
GraphPPL.factor_alias(
    ::ReactiveMPGraphPPLBackend,
    ::Type{Normal},
    ::GraphPPL.StaticInterfaces{(:μ, :τ)},
) = ExponentialFamily.NormalMeanPrecision
GraphPPL.default_parametrization(
    ::ReactiveMPGraphPPLBackend, ::GraphPPL.Atomic, ::Type{Normal}, rhs
) = error(
    "`Normal` cannot be constructed without keyword arguments. Use `Normal(mean = ..., var = ...)` or `Normal(mean = ..., precision = ...)`.",
)

# GraphPPL.interfaces(::ReactiveMPGraphPPLBackend, ::Type{<:ExponentialFamily.NormalMeanVariance}, _) = GraphPPL.StaticInterfaces((:out, :μ, :v))
# GraphPPL.interfaces(::ReactiveMPGraphPPLBackend, ::Type{<:ExponentialFamily.NormalMeanPrecision}, _) = GraphPPL.StaticInterfaces((:out, :μ, :τ))

GraphPPL.interface_aliases(::ReactiveMPGraphPPLBackend, ::Type{Normal}) =
    GraphPPL.StaticInterfaceAliases((
        (:mean, :μ),
        (:m, :μ),
        (:variance, :v),
        (:var, :v),
        (:τ⁻¹, :v),
        (:σ², :v),
        (:precision, :τ),
        (:prec, :τ),
        (:p, :τ),
        (:w, :τ),
        (:σ⁻², :τ),
        (:γ, :τ),
    ))

GraphPPL.factor_alias(
    ::ReactiveMPGraphPPLBackend,
    ::Type{MvNormal},
    ::GraphPPL.StaticInterfaces{(:μ, :Σ)},
) = ExponentialFamily.MvNormalMeanCovariance
GraphPPL.factor_alias(
    ::ReactiveMPGraphPPLBackend,
    ::Type{MvNormal},
    ::GraphPPL.StaticInterfaces{(:μ, :Λ)},
) = ExponentialFamily.MvNormalMeanPrecision
GraphPPL.default_parametrization(
    ::ReactiveMPGraphPPLBackend, ::GraphPPL.Atomic, ::Type{MvNormal}, rhs
) = error(
    "`MvNormal` cannot be constructed without keyword arguments. Use `MvNormal(mean = ..., covariance = ...)` or `MvNormal(mean = ..., precision = ...)`.",
)

GraphPPL.interface_aliases(::ReactiveMPGraphPPLBackend, ::Type{MvNormal}) =
    GraphPPL.StaticInterfaceAliases((
        (:mean, :μ),
        (:m, :μ),
        (:covariance, :Σ),
        (:cov, :Σ),
        (:Λ⁻¹, :Σ),
        (:V, :Σ),
        (:precision, :Λ),
        (:prec, :Λ),
        (:W, :Λ),
        (:Σ⁻¹, :Λ),
    ))

GraphPPL.factor_alias(
    ::ReactiveMPGraphPPLBackend,
    ::Type{Gamma},
    ::GraphPPL.StaticInterfaces{(:α, :θ)},
) = ExponentialFamily.GammaShapeScale
GraphPPL.factor_alias(
    ::ReactiveMPGraphPPLBackend,
    ::Type{Gamma},
    ::GraphPPL.StaticInterfaces{(:α, :β)},
) = ExponentialFamily.GammaShapeRate
GraphPPL.default_parametrization(
    backend::ReactiveMPGraphPPLBackend,
    nodetype::GraphPPL.Atomic,
    factor::Type{Gamma},
    rhs,
) = begin
    @warn "'Gamma' and 'GammaShapeScale' without keywords are constructed with parameters (Shape, Scale)." maxlog=1
    inputs = Base.tail(MessagePassingRulesBase.interfaces(factor))
    isequal(length(inputs), length(rhs)) || error("`$(factor)` has `$(length(inputs))` input interfaces `$(inputs)`, but `$(length(rhs))` arguments provided.")
    return NamedTuple{inputs}(rhs)
end

GraphPPL.interface_aliases(::ReactiveMPGraphPPLBackend, ::Type{Gamma}) =
    GraphPPL.StaticInterfaceAliases((
        (:a, :α),
        (:shape, :α),
        (:β⁻¹, :θ),
        (:scale, :θ),
        (:θ⁻¹, :β),
        (:rate, :β),
    ))
