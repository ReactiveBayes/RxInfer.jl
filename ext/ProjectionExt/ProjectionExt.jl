module ProjectionExt

using RxInfer, ExponentialFamily, ExponentialFamilyProjection, ReactiveMP, BayesBase, Random, LinearAlgebra

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
    (prodleft, prodright) = BayesBase.getleft(something), BayesBase.getright(something)
    return ReactiveMP.constrain_form(constraint::ProjectedTo, context, something, prodleft, prodright)
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, something::ProductOf, prodleft, prodright)
    return ReactiveMP.constrain_form(constraint::ProjectedTo, context, (x) -> logpdf(prodleft, x) + logpdf(prodright, x))
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

function ReactiveMP.constrain_form(constraint::ProjectedTo, context::ProjectionContext, something::TerminalProdArgument{S}) where { S <: SampleList }
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