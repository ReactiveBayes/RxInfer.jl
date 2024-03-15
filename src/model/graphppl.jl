import GraphPPL
import MacroTools
import ExponentialFamily

import MacroTools: @capture

"""
A backend for GraphPPL that uses ReactiveMP for inference.
"""
struct ReactiveMPGraphPPLBackend end

# Model specification with `@model` macro

function GraphPPL.model_macro_interior_pipelines(::ReactiveMPGraphPPLBackend)
    default_pipelines = GraphPPL.model_macro_interior_pipelines(GraphPPL.DefaultBackend())
    return (
        RxInfer.compose_simple_operators_with_brackets, 
        RxInfer.inject_tilderhs_aliases,
        default_pipelines...
    )
end

"""
    compose_simple_operators_with_brackets(expr::Expr)

This pipeline converts simple multi-argument operators to their corresponding bracketed expression. 
E.g. the expression `x ~ x1 + x2 + x3 + x4` becomes `x ~ ((x1 + x2) + x3) + x4)`.
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
        return recursive_brackets_expression(operator, vcat([Expr(:call, operator, args[1], args[2])], args[3:end]))
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

function inject_tilderhs_aliases(e::Expr)
    if @capture(e, lhs_ ~ rhs_)
        newrhs = MacroTools.postwalk(rhs) do expression
            # We short-circuit if `mflag` is true
            _expression, _ = foldl(ReactiveMPNodeAliases; init = (expression, false)) do (expression, mflag), alias
                return mflag ? (expression, true) : apply_alias_transformation(expression, alias)
            end
            return _expression
        end
        return :($lhs ~ $newrhs)
    else
        return e
    end
end

ReactiveMPNodeAliases = (
    (
        (expression) -> @capture(expression, a_ || b_) ? :(ReactiveMP.OR($a, $b)) : expression,
        "`a || b`: alias for `ReactiveMP.OR(a, b)` node (operator precedence between `||`, `&&`, `->` and `!` is the same as in Julia)."
    ),
    (
        (expression) -> @capture(expression, a_ && b_) ? :(ReactiveMP.AND($a, $b)) : expression,
        "`a && b`: alias for `ReactiveMP.AND(a, b)` node (operator precedence `||`, `&&`, `->` and `!` is the same as in Julia)."
    ),
    (
        (expression) -> @capture(expression, a_ -> b_) ? :(ReactiveMP.IMPLY($a, $b)) : expression,
        "`a -> b`: alias for `ReactiveMP.IMPLY(a, b)` node (operator precedence `||`, `&&`, `->` and `!` is the same as in Julia)."
    ),
    (
        (expression) -> @capture(expression, (¬a_) | (!a_)) ? :(ReactiveMP.NOT($a)) : expression,
        "`¬a` and `!a`: alias for `ReactiveMP.NOT(a)` node (Unicode `\\neg`, operator precedence `||`, `&&`, `->` and `!` is the same as in Julia)."
    )
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
See the documentation to `GraphPPL.@model` for more information.

## Supported aliases in the model specification specifically for RxInfer.jl and ReactiveMP.jl
$(begin io = IOBuffer(); RxInfer.show_tilderhs_alias(io); String(take!(io)) end)
"""
macro model(model_specification)
    return esc(GraphPPL.model_macro_interior(ReactiveMPGraphPPLBackend(), model_specification))
end

# Backend specific methods

function GraphPPL.NodeBehaviour(backend::ReactiveMPGraphPPLBackend, something::F) where {F}
    # Check the `sdtype` from `ReactiveMP` instead of using the `DefaultBackend`
    return GraphPPL.NodeBehaviour(backend, ReactiveMP.sdtype(something), something)
end
function GraphPPL.NodeBehaviour(backend::ReactiveMPGraphPPLBackend, ::ReactiveMP.Deterministic, _)
    return GraphPPL.Deterministic()
end
function GraphPPL.NodeBehaviour(backend::ReactiveMPGraphPPLBackend, ::ReactiveMP.Stochastic, _)
    return GraphPPL.Stochastic()
end

function GraphPPL.NodeType(::ReactiveMPGraphPPLBackend, something::F) where {F}
    # Fallback to the default behaviour
    return GraphPPL.NodeType(GraphPPL.DefaultBackend(), something)
end
function GraphPPL.aliases(::ReactiveMPGraphPPLBackend, something::F) where {F}
    # Fallback to the default behaviour
    return GraphPPL.aliases(GraphPPL.DefaultBackend(), something)
end

function GraphPPL.interfaces(backend::ReactiveMPGraphPPLBackend, something::F, ninputs) where {F}
    # Check `interfaces` from `ReactiveMP` and fallback to the `DefaultBackend` is those are `nothing`
    return GraphPPL.interfaces(backend, ReactiveMP.interfaces(something), something, ninputs)
end
function GraphPPL.interfaces(::ReactiveMPGraphPPLBackend, ::Val{I}, something, ninputs) where {I}
    if isequal(length(I), ninputs)
        return GraphPPL.StaticInterfaces(I)
    else
        error("`$(something)` has `$(length(I))` interfaces `$(I)`, but `$(ninputs)` requested.")
    end
end
function GraphPPL.interfaces(::ReactiveMPGraphPPLBackend, ::Nothing, something::F, ninputs) where {F}
    return GraphPPL.interfaces(GraphPPL.DefaultBackend(), something, ninputs)
end

function GraphPPL.factor_alias(::ReactiveMPGraphPPLBackend, something::F, interfaces) where {F}
    # Fallback to the default behaviour
    return GraphPPL.factor_alias(GraphPPL.DefaultBackend(), something, interfaces)
end
function GraphPPL.interface_aliases(::ReactiveMPGraphPPLBackend, something::F) where {F}
    # Fallback to the default behaviour
    return GraphPPL.interface_aliases(GraphPPL.DefaultBackend(), something)
end

function GraphPPL.default_parametrization(backend::ReactiveMPGraphPPLBackend, nodetype, something::F, rhs) where {F}
    # First check `inputinterfaces` from `ReacticeMP` and fallback to the `DefaultBackend` is those are `nothing`
    return GraphPPL.default_parametrization(backend, nodetype, ReactiveMP.inputinterfaces(something), something, rhs)
end
function GraphPPL.default_parametrization(backend::ReactiveMPGraphPPLBackend, ::GraphPPL.Atomic, ::Val{I}, something, rhs) where {I}
    if isequal(length(I), length(rhs))
        return NamedTuple{I}(rhs)
    else
        error("`$(something)` has `$(length(I))` input interfaces `$(I)`, but `$(length(rhs))` arguments provided.")
    end
end
function GraphPPL.default_parametrization(backend::ReactiveMPGraphPPLBackend, nodetype, ::Nothing, something::F, rhs) where {F}
    return GraphPPL.default_parametrization(GraphPPL.DefaultBackend(), nodetype, something, rhs)
end

# Node specific aliases

GraphPPL.factor_alias(::ReactiveMPGraphPPLBackend, ::Type{Normal}, ::GraphPPL.StaticInterfaces{(:μ, :v)}) = ExponentialFamily.NormalMeanVariance
GraphPPL.factor_alias(::ReactiveMPGraphPPLBackend, ::Type{Normal}, ::GraphPPL.StaticInterfaces{(:μ, :τ)}) = ExponentialFamily.NormalMeanPrecision
GraphPPL.default_parametrization(::ReactiveMPGraphPPLBackend, ::Type{Normal}) =
    error("`Normal` cannot be constructed without keyword arguments. Use `Normal(mean = ..., var = ...)` or `Normal(mean = ..., precision = ...)`.")

# GraphPPL.interfaces(::ReactiveMPGraphPPLBackend, ::Type{<:ExponentialFamily.NormalMeanVariance}, _) = GraphPPL.StaticInterfaces((:out, :μ, :v))
# GraphPPL.interfaces(::ReactiveMPGraphPPLBackend, ::Type{<:ExponentialFamily.NormalMeanPrecision}, _) = GraphPPL.StaticInterfaces((:out, :μ, :τ))

GraphPPL.interface_aliases(::ReactiveMPGraphPPLBackend, ::Type{Normal}) = GraphPPL.StaticInterfaceAliases((
    (:mean, :μ), (:m, :μ), (:variance, :v), (:var, :v), (:τ⁻¹, :v), (:σ², :v), (:precision, :τ), (:prec, :τ), (:p, :τ), (:w, :τ), (:σ⁻², :τ), (:γ, :τ)
))

GraphPPL.factor_alias(::ReactiveMPGraphPPLBackend, ::Type{MvNormal}, ::GraphPPL.StaticInterfaces{(:μ, :Σ)}) = ExponentialFamily.MvNormalMeanCovariance
GraphPPL.factor_alias(::ReactiveMPGraphPPLBackend, ::Type{MvNormal}, ::GraphPPL.StaticInterfaces{(:μ, :Λ)}) = ExponentialFamily.MvNormalMeanPrecision
GraphPPL.default_parametrization(::ReactiveMPGraphPPLBackend, ::Type{MvNormal}) =
    error("`MvNormal` cannot be constructed without keyword arguments. Use `MvNormal(mean = ..., covariance = ...)` or `MvNormal(mean = ..., precision = ...)`.")

GraphPPL.interface_aliases(::ReactiveMPGraphPPLBackend, ::Type{MvNormal}) =
    GraphPPL.StaticInterfaceAliases(((:mean, :μ), (:m, :μ), (:covariance, :Σ), (:cov, :Σ), (:Λ⁻¹, :Σ), (:V, :Σ), (:precision, :Λ), (:prec, :Λ), (:W, :Λ), (:Σ⁻¹, :Λ)))

GraphPPL.factor_alias(::ReactiveMPGraphPPLBackend, ::Type{Gamma}, ::GraphPPL.StaticInterfaces{(:α, :θ)}) = ExponentialFamily.GammaShapeScale
GraphPPL.factor_alias(::ReactiveMPGraphPPLBackend, ::Type{Gamma}, ::GraphPPL.StaticInterfaces{(:α, :β)}) = ExponentialFamily.GammaShapeRate
GraphPPL.default_parametrization(::ReactiveMPGraphPPLBackend, ::Type{Gamma}) =
    error("`Gamma` cannot be constructed without keyword arguments. Use `Gamma(shape = ..., rate = ...)` or `Gamma(shape = ..., scale = ...)`.")

# GraphPPL.interfaces(::ReactiveMPGraphPPLBackend, ::Type{<:ExponentialFamily.GammaShapeScale}, _) = GraphPPL.StaticInterfaces((:out, :α, :θ))
# GraphPPL.interfaces(::ReactiveMPGraphPPLBackend, ::Type{<:ExponentialFamily.GammaShapeRate}, _) = GraphPPL.StaticInterfaces((:out, :α, :β))

GraphPPL.interface_aliases(::ReactiveMPGraphPPLBackend, ::Type{Gamma}) =
    GraphPPL.StaticInterfaceAliases(((:a, :α), (:shape, :α), (:β⁻¹, :θ), (:scale, :θ), (:θ⁻¹, :β), (:rate, :β)))
