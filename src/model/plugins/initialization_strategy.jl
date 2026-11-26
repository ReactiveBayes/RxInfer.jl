export FromPriors

"""
    InitializationStrategy

The abstract type of strategies that initialize the variables an initialization specification
leaves out. A strategy is passed to `infer` as `initialization`, and
[`RxInfer.initial_marginal`](@ref)`(strategy, model, label)` is called for every random variable
without an initial marginal. The specification of the strategy,
[`RxInfer.initialization_specification`](@ref), is applied first, and the strategy does not
overwrite it.

See [`FromPriors`](@ref).
"""
abstract type InitializationStrategy end

"""
    RxInfer.initial_marginal(strategy::InitializationStrategy, model, label)

Return the initial marginal of the random variable `label` of the GraphPPL `model`, or `nothing`
to leave it uninitialized. Called once for each random variable without an initial marginal, in
the order in which the model created them.
"""
function initial_marginal end

"""
    RxInfer.initialization_specification(strategy::InitializationStrategy)

Return the `@initialization` specification that `strategy` applies before it initializes the
remaining variables, or `nothing` for the model's default.
"""
initialization_specification(::InitializationStrategy) = nothing

function GraphPPL.postprocess_plugin(
        plugin::InitializationPlugin{<:InitializationStrategy}, model::Model
    )
    strategy = plugin.initialization
    specification = initialization_specification(strategy)
    apply_init!(
        model,
        something(
            specification, default_init(GraphPPL.fform(GraphPPL.getcontext(model)))
        ),
    )
    variable_nodes(model) do label, nodedata
        is_random(getproperties(nodedata)) || return nothing
        hasextra(nodedata, InitMarExtraKey) && return nothing
        marginal = initial_marginal(strategy, model, label)
        if !isnothing(marginal) && !hasextra(nodedata, InitMarExtraKey)
            setextra!(nodedata, InitMarExtraKey, marginal)
        end
        return nothing
    end
    return nothing
end

"""
    FromPriors(specification = nothing)

An [`InitializationStrategy`](@ref) that initializes each random variable that `specification`,
an [`@initialization`](@ref) block, leaves out from its prior: the distribution of the node whose
`out` the variable is, evaluated at the initial means of its parents. A constant parent contributes
its value, and a random parent contributes the mean of its own initial marginal, given or from its
prior, so chains and hierarchies are initialized from the root down. A deterministic parent,
`z := f(x)`, contributes `f` evaluated at the means of its inputs, and is not initialized itself.

Only marginals are set, never messages, so belief propagation is unchanged. A variable whose prior
cannot be formed is left uninitialized: a variable with no distribution node on its `out`, with a
data parent, or with parameters that its distribution does not accept.

The components of a mixture have equal priors, and therefore equal initial marginals; break the
symmetry in `specification`.

```julia
infer(model = m(), data = (y = y,), initialization = FromPriors())
infer(model = m(), data = (y = y,), initialization = FromPriors(@initialization begin
    q(μ) = [NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(1.0, 1.0)]
end))
```
"""
struct FromPriors{S} <: InitializationStrategy
    specification::S
end

FromPriors() = FromPriors(nothing)

initialization_specification(strategy::FromPriors) = strategy.specification

function Base.show(io::IO, strategy::FromPriors)
    print(io, "FromPriors(")
    isnothing(strategy.specification) || print(io, "…")
    return print(io, ")")
end

function initial_marginal(::FromPriors, model::Model, label::NodeLabel)
    nodedata = model[label]
    hasextra(nodedata, InitMarExtraKey) &&
        return getextra(nodedata, InitMarExtraKey)
    prior = variable_prior(model, label)
    # a deterministic node's output is not initialised: its value is its children's business
    prior isa Distribution || return nothing
    setextra!(nodedata, InitMarExtraKey, prior)
    return prior
end

# The prior of the random variable `label` at its parents' initial means, as `prior_at_means`
# gives it, computed once per variable. A prior that cannot be formed is remembered too, so a chain
# whose root has none is walked once, not once per descendant.
const FromPriorsExtraKey = GraphPPL.NodeDataExtraKey{:from_priors, Any}()

function variable_prior(model::Model, label::NodeLabel)
    nodedata = model[label]
    hasextra(nodedata, FromPriorsExtraKey) &&
        return getextra(nodedata, FromPriorsExtraKey)
    parent = out_factor(model, label)
    prior = isnothing(parent) ? nothing :
        prior_at_means(model, getproperties(model[parent]))
    setextra!(nodedata, FromPriorsExtraKey, prior)
    return prior
end

# The value of a deterministic node at its inputs' means, which its children read and nothing
# initialises
struct PriorValue{T}
    value::T
end

# The factor node whose `out` the variable `label` is, or `nothing`
function out_factor(model::Model, label::NodeLabel)
    for factor in GraphPPL.Graphs.neighbors(model, label)
        for (variable, edge, _) in GraphPPL.neighbors(getproperties(model[factor]))
            variable == label && GraphPPL.getname(edge) === :out &&
                return factor
        end
    end
    return nothing
end

# The distribution of a node at its parents' initial means; a `PriorValue` for a deterministic
# node; `nothing` when it cannot be formed
function prior_at_means(model::Model, properties::FactorNodeProperties)
    fform = GraphPPL.fform(properties)
    # `x ~ d` for a distribution value
    fform isa Distribution && return fform
    inputs = prior_inputs(model, properties)
    isnothing(inputs) && return nothing
    deterministic =
        isdeclarednode(fform) ?
        MessagePassingRulesBase.sdtype(fform) isa
        MessagePassingRulesBase.Deterministic : fform isa Function
    try
        value = fform(inputs...)
        return deterministic ? PriorValue(value) :
            (value isa Distribution ? value : nothing)
    catch error
        error isa Union{MethodError, ArgumentError, DomainError} || rethrow()
        return nothing
    end
end

# The means of a node's inputs in the order its constructor takes them: the declared interfaces
# after `out`, members of a group by their index; `nothing` when one has no mean yet
function prior_inputs(model::Model, properties::FactorNodeProperties)
    fform = GraphPPL.fform(properties)
    edges = [
        (edge, label) for (label, edge, _) in GraphPPL.neighbors(properties) if
            GraphPPL.getname(edge) !== :out
    ]
    order = if isdeclarednode(fform)
        names = Base.tail(Tuple(MessagePassingRulesBase.interfaces(fform)))
        isempty(MessagePassingRulesBase.interface_groups(fform)) || return nothing
        all(edge -> GraphPPL.getname(first(edge)) in names, edges) ||
            return nothing
        sort(edges; by = edge -> findfirst(isequal(GraphPPL.getname(first(edge))), names))
    else
        sort(edges; by = edge -> something(first(edge).index, 1))
    end
    values = map(((_, label),) -> initial_mean(model, label), order)
    any(isnothing, values) && return nothing
    return values
end

# A parent's value for its children's priors: a constant's value, a random variable's initial
# mean, a deterministic node's value at its inputs' means
function initial_mean(model::Model, label::NodeLabel)
    properties = getproperties(model[label])
    is_constant(properties) && return GraphPPL.value(properties)
    is_random(properties) || return nothing
    marginal = initial_marginal(FromPriors(), model, label)
    isnothing(marginal) || return mean(marginal)
    prior = variable_prior(model, label)
    return prior isa PriorValue ? prior.value : nothing
end
