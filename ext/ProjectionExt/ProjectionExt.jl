module ProjectionExt

using RxInfer, ExponentialFamily, ExponentialFamilyProjection, ReactiveMP, BayesBase, Random, LinearAlgebra

ReactiveMP.default_form_check_strategy(::ProjectedTo) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::ProjectedTo) = GenericProd()

mutable struct ProjectionContext{T}
    previous::Union{Nothing, T}
end

function ReactiveMP.prepare_context(constraint::ProjectedTo)
    T = ExponentialFamilyProjection.get_projected_to_type(constraint)
    return ProjectionContext{T}(nothing)
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, something::Union{Distribution, ExponentialFamilyDistribution})
    T = ExponentialFamilyProjection.get_projected_to_type(constraint)
    D = ExponentialFamily.exponential_family_typetag(something)
    if T === D
        result = convert(D, something)
        context.previous = result
        return result
    else
        return ReactiveMP.constrain_form(constraint, context, (x) -> logpdf(something, x))
    end
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, something::ProductOf)
    return ReactiveMP.constrain_form(constraint::ProjectedTo, context, (x) -> logpdf(something, x))
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, fn)
    initialpoint = nothing
    if !isnothing(context.previous)
        initialη = getnaturalparameters(convert(ExponentialFamilyDistribution, context.previous))
        manifold = ExponentialFamilyProjection.get_projected_to_manifold(constraint)
        initialpoint = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(manifold, initialη)
    end
    result = project_to(constraint, fn; initialpoint = initialpoint)
    context.previous = result
    return result
end

end