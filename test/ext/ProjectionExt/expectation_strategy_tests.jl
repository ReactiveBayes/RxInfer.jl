# `ProjectedTo` with ExponentialFamilyProjection's `ExpectationStrategy`: the projection to a family
# beyond the Gaussian by quadrature, deterministic, reporting its evidence lower bound.

@testitem "ProjectedTo(Beta) by quadrature: a conjugate posterior through a rule-free likelihood" begin
    using RxInfer, BayesBase, Distributions, ExponentialFamilyProjection, SpecialFunctions
    const EA = Base.require(Base.PkgId(Base.UUID("9cdca625-6a81-42ba-9f59-d0fb182fc17f"), "ExpectationApproximations"))

    # a coin's flip with no rules: the fallback sends its log-density
    struct Flip{P} <: DiscreteUnivariateDistribution
        p::P
    end
    BayesBase.logpdf(d::Flip, y) = y * log(d.p) + (1 - y) * log1p(-d.p)
    BayesBase.insupport(::Flip, y) = true
    @define_factor_node(node = Flip, type = Stochastic, interfaces = [:out, :p])

    @model function coin(y)
        p ~ Beta(2.0, 2.0)
        y .~ Flip(p)
    end

    flips = Float64[1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    k, n = count(isone, flips), length(flips)
    strategy = ExpectationStrategy(EA.EADefault())
    projection = ProjectedTo(Beta; parameters = ProjectionParameters(strategy = strategy, niterations = 2000, tolerance = 1.0e-10))
    result = infer(
        model = coin(), data = (y = flips,), fallback = LaplaceApproximation(),
        options = (default_form_constraint = (q = projection,),), report = (:approximations,), logscales = true,
    )
    q = result.posteriors[:p]
    # the family holds the posterior: Beta(2 + k, 2 + n - k)
    @test all(isapprox.(params(q), (2.0 + k, 2.0 + n - k); rtol = 1.0e-3))
    # its bound is then the log evidence, log B(2 + k, 2 + n - k) - log B(2, 2)
    p = result.report.approximations.variables[(:p, nothing)]
    @test p.marginal.outcome.method === :projection && p.marginal.outcome.family === :Beta
    @test p.marginal.outcome.logZ ≈ logbeta(2.0 + k, 2.0 + n - k) - logbeta(2.0, 2.0) rtol = 1.0e-3
    @test startswith(first(report_tables(result.report.approximations)).groups[1].second, "p: projected onto Beta, ELBO = ")
end
