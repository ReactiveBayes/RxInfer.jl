module ProjectionExt

using RxInfer, ExponentialFamily, ExponentialFamilyProjection, ReactiveMP, BayesBase, Random, LinearAlgebra

ReactiveMP.default_form_check_strategy(::ProjectedTo) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::ProjectedTo) = GenericProd()
ReactiveMP.default_form_check_strategy(::Union{Vector{<:ProjectedTo}, Tuple}) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::Union{Vector{<:ProjectedTo}, Tuple})  = GenericProd()

mutable struct ProjectionContext{T}
    previous::Union{Nothing, T}
end

function ReactiveMP.prepare_context(constraint::ProjectedTo)
    T = ExponentialFamilyProjection.get_projected_to_type(constraint)
    return ProjectionContext{T}(nothing)
end

function ReactiveMP.prepare_context(constraint::Union{Vector{<:ProjectedTo}, Tuple}) 
    T = map(ExponentialFamilyProjection.get_projected_to_type, constraint)
    return map(t -> ProjectionContext{t}(nothing), T)
end

function initial_point_from_context_constraint(previous_context, constraint)
    if !isnothing(previous_context)
        initialη = getnaturalparameters(convert(ExponentialFamilyDistribution, previous_context))
        manifold = ExponentialFamilyProjection.get_projected_to_manifold(constraint)
        initialpoint = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(manifold, initialη)

        return initialpoint

    else
        return nothing
    end
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, something::Distribution)
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
    return ReactiveMP.constrain_form(constraint::ProjectedTo, context, (x) -> logpdf(something, x) )
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, fn)
    initialpoint = initial_point_from_context_constraint(context.previous, constraint)
    result = project_to(constraint, fn; initialpoint = initialpoint)
    context.previous = result
    return result
end

function ReactiveMP.constrain_form(constraints::Vector{<:ProjectedTo}, contexts::Vector{<:ProjectionContext}, fn::Base.Generator) 
    length_constraints = length(constraints)
    result = map((f, constraint, context) -> project_to(constraint, x -> logpdf(f, x); initialpoint =initial_point_from_context_constraint(context.previous, constraint) ) , fn, constraints, contexts)
    map((r,c) -> c.previous = r , result, contexts)
    return FactorizedJoint(ntuple(i -> result[i] , length_constraints))
end


end