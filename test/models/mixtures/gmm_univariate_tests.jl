@testitem "Univariate Gaussian Mixture model " begin
    using BenchmarkTools, Plots, LinearAlgebra, StableRNGs

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    @model function univariate_gaussian_mixture_model(y)
        s ~ Beta(1.0, 1.0)

        m[1] ~ Normal(mean = -2.0, variance = 1e3)
        p[1] ~ Gamma(shape = 0.01, rate = 0.01)

        m[2] ~ Normal(mean = 2.0, variance = 1e3)
        p[2] ~ Gamma(shape = 0.01, rate = 0.01)

        for i in eachindex(y)
            z[i] ~ Bernoulli(s)
            y[i] ~ NormalMixture(switch = z[i], m = m, p = p)
        end
    end

    init = @initialization begin
        q(s) = vague(Beta)
        q(m) = [NormalMeanVariance(-2.0, 1e3), NormalMeanVariance(2.0, 1e3)]
        q(p) = [vague(GammaShapeRate), vague(GammaShapeRate)]
    end

    function inference_univariate(data, n_its, constraints)
        return infer(
            model          = univariate_gaussian_mixture_model(),
            data           = (y = data,),
            constraints    = constraints,
            returnvars     = KeepEach(),
            free_energy    = Float64,
            iterations     = n_its,
            initialization = init
        )
    end
    ## -------------------------------------------- ##
    ## Data creation
    ## -------------------------------------------- ##
    n = 150

    rng = StableRNG(12345)

    switch = [1 / 3, 2 / 3]

    μ1 = -10.0
    μ2 = 10.0
    w  = 1.777

    z = rand(rng, Categorical(switch), n)
    y = Vector{Float64}(undef, n)

    dists = [Normal(μ1, sqrt(inv(w))), Normal(μ2, sqrt(inv(w)))]

    for i in 1:n
        y[i] = rand(rng, dists[z[i]])
    end

    ## Inference execution
    constraints = @constraints begin
        q(z, s, m, p) = q(z)q(s)q(m)q(p)
        q(m) = q(m[begin]) .. q(m[end])
        q(p) = q(p[begin]) .. q(p[end])
    end

    # Execute inference for different constraints specifications
    results = map((constraints) -> inference_univariate(y, 10, constraints), [MeanField(), constraints])

    fresult = results[begin]
    # All execution must be equivalent (check against first)
    foreach(results[(begin + 1):end]) do result
        foreach(zip(fresult.posteriors, result.posteriors)) do (l, r)
            @test l == r
        end
    end

    mswitch = fresult.posteriors[:s]
    mm1 = getindex.(fresult.posteriors[:m], 1)
    mm2 = getindex.(fresult.posteriors[:m], 2)
    mw1 = getindex.(fresult.posteriors[:p], 1)
    mw2 = getindex.(fresult.posteriors[:p], 2)
    fe = fresult.free_energy

    # Test inference results
    @test length(mswitch) === 10
    @test length(mm1) === 10
    @test length(mm2) === 10
    @test length(mw1) === 10
    @test length(mw2) === 10
    @test length(fe) === 10 && all(filter(e -> abs(e) > 1e-3, diff(fe)) .< 0)
    @test abs(last(fe) - 139.74362) < 0.01

    ms = mean(last(mswitch))

    @test ((abs(ms - switch[1]) < 0.1) || (abs(ms - switch[2]) < 0.1))

    ems = sort([last(mm1), last(mm2)], by = mean)
    rms = sort([μ1, μ2])

    foreach(zip(rms, ems)) do (r, e)
        @test abs(r - mean(e)) < 0.19
    end

    ews = sort([last(mw1), last(mw2)], by = mean)
    rws = sort([w, w])

    foreach(zip(rws, ews)) do (r, e)
        @test abs(r - mean(e)) < 0.15
    end

    @test_throws "must be the naive mean-field" inference_univariate(y, 10, BetheFactorization())
    @test_throws "must be the naive mean-field" inference_univariate(y, 10, @constraints(
        begin end
    ))

    @test_plot "models" "gmm_univariate" begin
        dim(d) = (a) -> map(r -> r[d], a)
        mp = plot(mean.(mm1), ribbon = var.(mm1) .|> sqrt, label = "m1 prediction")
        mp = plot!(mean.(mm2), ribbon = var.(mm2) .|> sqrt, label = "m2 prediction")
        mp = plot!(mp, [μ1], seriestype = :hline, label = "real m1")
        mp = plot!(mp, [μ2], seriestype = :hline, label = "real m2")

        wp = plot(mean.(mw1), ribbon = var.(mw1) .|> sqrt, label = "w1 prediction", legend = :bottomleft, ylim = (-1, 3))
        wp = plot!(wp, [w], seriestype = :hline, label = "real w1")
        wp = plot!(wp, mean.(mw2), ribbon = var.(mw2) .|> sqrt, label = "w2 prediction")
        wp = plot!(wp, [w], seriestype = :hline, label = "real w2")

        swp = plot(mean.(mswitch), ribbon = var.(mswitch) .|> sqrt, label = "Switch prediction")

        swp = plot!(swp, [switch[1]], seriestype = :hline, label = "switch[1]")
        swp = plot!(swp, [switch[2]], seriestype = :hline, label = "switch[2]")

        fep = plot(fe[2:end], label = "Free Energy", legend = :bottomleft)

        return plot(mp, wp, swp, fep, layout = @layout([a b; c d]), size = (1000, 700))
    end

    @test_benchmark "models" "gmm_univariate" inference_univariate($y, 10, MeanField())
end
