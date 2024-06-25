module ProjectionExt

using RxInfer, ExponentialFamilyProjection, ReactiveMP, BayesBase

ReactiveMP.default_form_check_strategy(::ProjectedTo) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::ProjectedTo) = GenericProd()

function ReactiveMP.constrain_form(constraint::ProjectedTo, something::ProductOf)
    return ReactiveMP.constrain_form(constraint::ProjectedTo, BayesBase.getleft(something), BayesBase.getright(something))
end

function ReactiveMP.constrain_form(constraint::ProjectedTo, prior, message::LinearizedProductOf)
    error(1)
    initialpoint = prior
    batch        = 10
    k            = 1

    T = ExponentialFamilyProjection.get_projected_to_type(constraint)
    M = ExponentialFamilyProjection.get_projected_to_manifold(constraint)

    r = Random.randperm(StableRNG(42), length(message))

    while k < length(message)
        v = (k:min(k + batch - 1, length(message)))
        m = LinearizedProductOf(message.vector[r[v]], batch)
        k = (k + batch)
        initialpointp = if initialpoint isa T
            ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(M, getnaturalparameters(convert(ExponentialFamilyDistribution, initialpoint)))
        else
            rand(StableRNG(42), M)
        end
        q = project_to(constraint, (x) -> logpdf(initialpoint, x) + logpdf(m, x); initialpoint = initialpointp)
        initialpoint = q
        if norm(var(initialpoint)) < 1e-3
            return initialpoint
        end
    end

    return initialpoint
end

end