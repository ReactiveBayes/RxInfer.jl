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
    return ReactiveMP.constrain_form(constraint::ProjectedTo, context, (x) -> logpdf(something, x))
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, fn::F) where {F <: Function}
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

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, something::TerminalProdArgument{S}) where {S <: SampleList}
    arg = something.argument

    f = (M, p) -> begin
        ef = convert(ExponentialFamilyDistribution, M, p)
        return -sum((d) -> logpdf(ef, d), BayesBase.get_samples(arg))
    end

    g = (M, p) -> begin
        ef = convert(ExponentialFamilyDistribution, M, p)
        X = ReactiveMP.ForwardDiff.gradient((p) -> f(M, p), p)
        inv_fisher = cholinv(ExponentialFamily.fisherinformation(ef))
        X = inv_fisher * X
        X = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(M, X)
        X = X ./ norm(M, p, X)
        return X
    end

    M = ExponentialFamilyProjection.get_projected_to_manifold(constraint)
    p = rand(MersenneTwister(42), M)
    q = ExponentialFamilyProjection.Manopt.gradient_descent(M, f, g, p)

    return convert(Distribution, convert(ExponentialFamilyDistribution, M, q))

    # (prodleft, prodright) = BayesBase.getleft(something), BayesBase.getright(something)
    # return ReactiveMP.constrain_form(constraint::ProjectedTo, context, something, prodleft, prodright)
end

end