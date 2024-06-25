module ProjectionExt

using RxInfer, ExponentialFamily, ExponentialFamilyProjection, ReactiveMP, BayesBase

ReactiveMP.default_form_check_strategy(::ProjectedTo) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::ProjectedTo) = GenericProd()

mutable struct ProjectionContext
    previous
end

function ReactiveMP.prepare_context(::ProjectedTo)
    return ProjectionContext(nothing)
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, something::Distribution)
    return ReactiveMP.constrain_form(constraint, context, ExponentialFamilyProjection.get_projected_to_type(constraint), something)
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, something::ProductOf)
    (left, right) = BayesBase.getleft(something), BayesBase.getright(something)
    return ReactiveMP.constrain_form(constraint::ProjectedTo, context, (x) -> logpdf(left, x) + logpdf(right, x))
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, ::Type{T}, something::T) where {T}
    return something
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, ::Type{R}, something::T) where {R, T}
    return ReactiveMP.constrain_form(constraint, context, (x) -> logpdf(something, x))
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, fn::F) where {F <: Function}
    initialpoint = nothing
    if context.previous !== nothing
        initialη = getnaturalparameters(convert(ExponentialFamilyDistribution, context.previous))
        manifold = ExponentialFamilyProjection.get_projected_to_manifold(constraint)
        initialpoint = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(manifold, initialη)
    end
    approximation = project_to(constraint, fn; initialpoint = initialpoint)
    context.previous = approximation
    return approximation
end

end