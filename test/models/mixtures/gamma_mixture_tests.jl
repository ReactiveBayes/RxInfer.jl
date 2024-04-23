@testitem "Gamma mixture model" begin
    using Distributions
    using BenchmarkTools, LinearAlgebra, StableRNGs, Plots

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    @model function gamma_mixture_model(y, nmixtures, priors_as, priors_bs, prior_s)

        # set prior on global selection variable
        s ~ Dirichlet(probvec(prior_s))

        # allocate vectors of random variables
        local as
        local bs

        # set priors on variables of mixtures
        for i in 1:nmixtures
            as[i] ~ Gamma(shape = shape(priors_as[i]), rate = rate(priors_as[i]))
            bs[i] ~ Gamma(shape = shape(priors_bs[i]), rate = rate(priors_bs[i]))
        end

        # specify local selection variable and data generating process
        for i in eachindex(y)
            z[i] ~ Categorical(s)
            y[i] ~ GammaMixture(switch = z[i], a = as, b = bs)
        end
    end

    constraints = @constraints begin
        q(z, as, bs, s) = q(z)q(as)q(bs)q(s)

        q(as) = q(as[begin]) .. q(as[end])
        q(bs) = q(bs[begin]) .. q(bs[end])

        q(as)::PointMassFormConstraint(starting_point = (args...) -> [1.0])
    end

    # specify seed and number of data points
    rng = StableRNG(43)
    n_samples = 250

    # specify parameters of mixture model that generates the data
    # Note that mixture components have exactly the same means
    mixtures  = [Gamma(9.0, inv(27.0)), Gamma(90.0, inv(270.0))]
    nmixtures = length(mixtures)
    mixing    = rand(rng, nmixtures)
    mixing    = mixing ./ sum(mixing)
    mixture   = MixtureModel(mixtures, mixing)

    # generate data set
    dataset = rand(rng, mixture, n_samples)

    priors_as = [Gamma(1.0, 0.1), Gamma(1.0, 1.0)]
    priors_bs = [Gamma(10.0, 2.0), Gamma(1.0, 3.0)]
    prior_s = Dirichlet(1e3 * mixing)

    gmodel      = gamma_mixture_model(nmixtures = nmixtures, priors_as = priors_as, priors_bs = priors_bs, prior_s = prior_s)
    gdata       = (y = dataset,)
    ginit       = @initialization begin
        q(s) = prior_s
        q(z) = vague(Categorical, nmixtures)
        q(bs) = GammaShapeRate(1.0, 1.0)
    end
    greturnvars = (s = KeepLast(), z = KeepLast(), as = KeepEach(), bs = KeepEach())

    gresult = infer(model = gmodel, data = gdata, constraints = constraints, initialization = ginit, returnvars = greturnvars, free_energy = true, iterations = 50)

    # extract inferred parameters
    _as, _bs = mean.(gresult.posteriors[:as][end]), mean.(gresult.posteriors[:bs][end])
    _dists   = map(g -> Gamma(g[1], inv(g[2])), zip(_as, _bs))
    _mixing  = mean(gresult.posteriors[:s])

    # create model from inferred parameters
    _mixture = MixtureModel(_dists, _mixing)

    @test mean(_dists[1]) ≈ 0.32 atol = 1e-2
    @test mean(_dists[2]) ≈ 0.33 atol = 1e-2
    @test _mixing ≈ [0.8, 0.2] atol = 1e-2
    @test last(gresult.free_energy) ≈ -146.8 atol = 1e-1

    @test_plot "models" "gamma_mixture" begin
        # plot results
        p1 = histogram(dataset, ylim = (0, 13), xlim = (0, 1), normalize = :pdf, label = "data", title = "Generated mixtures", opacity = 0.3)
        p1 = plot!(range(0.0, 1.0, length = 100), (x) -> mixing[1] * pdf(mixtures[1], x), label = "component 1", linewidth = 3.0)
        p1 = plot!(range(0.0, 1.0, length = 100), (x) -> mixing[2] * pdf(mixtures[2], x), label = "component 2", linewidth = 3.0)

        p2 = histogram(dataset, ylim = (0, 13), xlim = (0, 1), normalize = :pdf, label = "data", title = "Inferred mixtures", opacity = 0.3)
        p2 = plot!(range(0.0, 1.0, length = 100), (x) -> _mixing[1] * pdf(_dists[1], x), label = "component 1", linewidth = 3.0)
        p2 = plot!(range(0.0, 1.0, length = 100), (x) -> _mixing[2] * pdf(_dists[2], x), label = "component 2", linewidth = 3.0)

        # evaluate the convergence of the algorithm by monitoring the BFE
        p3 = plot(gresult.free_energy, label = false, xlabel = "iterations", title = "Bethe FE")

        plot(plot(p1, p2, layout = @layout([a; b])), plot(p3), layout = @layout([a b]), size = (800, 400))
    end
end
