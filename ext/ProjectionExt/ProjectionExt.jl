module ProjectionExt

using RxInfer,
    ExponentialFamily,
    ExponentialFamilyProjection,
    ReactiveMP,
    BayesBase,
    Random,
    LinearAlgebra

ReactiveMP.default_form_check_strategy(::ProjectedTo) =
    FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::ProjectedTo) = GenericProd()

mutable struct ProjectionContext{T}
    previous::Union{Nothing, T}
    # the evidence lower bound of the last projection, where its strategy computes expectations
    # deterministically, `nothing` otherwise
    logZ::Any
end

function ReactiveMP.prepare_context(constraint::ProjectedTo)
    T = ExponentialFamilyProjection.get_projected_to_type(constraint)
    return ProjectionContext{T}(nothing, nothing)
end

# The bound E_q[log p̃] + H[q] by the strategy's own backend: an `ExpectationStrategy` has one, a
# sampling strategy reports none
projection_bound(constraint::ProjectedTo, q, fn) = projection_bound(constraint.parameters.strategy, q, fn)
function projection_bound(strategy::ExpectationStrategy, q, fn)
    chosen = BayesBase.which_expectation_backend(strategy.backend, Any, typeof(q))
    isnothing(chosen) && return nothing
    return BayesBase.expectation(BayesBase.prepare_expectation(chosen, q), fn) + entropy(q)
end
projection_bound(strategy, q, fn) = nothing

function ReactiveMP.constrain_form_logscale(constraint::ProjectedTo, context::ProjectionContext, something)
    result = ReactiveMP.constrain_form(constraint, context, something)
    return result, context.logZ
end

RxInfer.form_outcome(constraint::ProjectedTo, context::ProjectionContext, distribution, result) =
    (method = :projection, family = nameof(ExponentialFamilyProjection.get_projected_to_type(constraint)), logZ = context.logZ)

function ReactiveMP.constrain_form(
        constraint::ProjectedTo,
        context::ProjectionContext,
        something::Union{Distribution, ExponentialFamilyDistribution},
    )
    T = ExponentialFamilyProjection.get_projected_to_type(constraint)
    D = ExponentialFamily.exponential_family_typetag(something)
    if T === D
        result = convert(D, something)
        context.previous = result
        context.logZ = nothing
        return result
    else
        return ReactiveMP.constrain_form(
            constraint, context, (x) -> logpdf(something, x)
        )
    end
end

function ReactiveMP.constrain_form(
        constraint::ProjectedTo, context::ProjectionContext, something::ProductOf
    )
    return ReactiveMP.constrain_form(
        constraint::ProjectedTo, context, (x) -> logpdf(something, x)
    )
end

function ReactiveMP.constrain_form(
        constraint::ProjectedTo, context::ProjectionContext, fn
    )
    initialpoint = nothing
    if !isnothing(context.previous)
        initialη = getnaturalparameters(
            convert(ExponentialFamilyDistribution, context.previous)
        )
        manifold = ExponentialFamilyProjection.get_projected_to_manifold(
            constraint
        )
        initialpoint = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(
            manifold, initialη
        )
    end
    result = project_to(constraint, fn; initialpoint = initialpoint)
    context.previous = result
    context.logZ = projection_bound(constraint, result, fn)
    return result
end

end
