module RxInferModelsGMMTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

## Model definition
## -------------------------------------------- ##
@model function univariate_gaussian_mixture_model(n)
    s ~ Beta(1.0, 1.0)

    m1 ~ NormalMeanVariance(-2.0, 1e3)
    w1 ~ GammaShapeRate(0.01, 0.01)

    m2 ~ NormalMeanVariance(2.0, 1e3)
    w2 ~ GammaShapeRate(0.01, 0.01)

    z = randomvar(n)
    y = datavar(Float64, n)

    for i in 1:n
        z[i] ~ Bernoulli(s)
        y[i] ~ NormalMixture(z[i], (m1, m2), (w1, w2))
    end

    return s, m1, w1, m2, w2, z, y
end

@model function multivariate_gaussian_mixture_model(rng, L, nmixtures, n)
    z = randomvar(n)
    m = randomvar(nmixtures)
    w = randomvar(nmixtures)

    basis_v = [1.0, 0.0]

    for i in 1:nmixtures
        # Assume we now only approximate location of cluters's mean
        approximate_angle_prior = ((2π + +rand(rng)) / nmixtures) * (i - 1)
        approximate_basis_v = L / 2 * (basis_v .+ rand(rng, 2))
        approximate_rotation = [
            cos(approximate_angle_prior) -sin(approximate_angle_prior)
            sin(approximate_angle_prior) cos(approximate_angle_prior)
        ]
        mean_mean_prior = approximate_rotation * approximate_basis_v
        mean_mean_cov = [1e6 0.0; 0.0 1e6]

        m[i] ~ MvNormalMeanCovariance(mean_mean_prior, mean_mean_cov)
        w[i] ~ Wishart(3, [1e2 0.0; 0.0 1e2])
    end

    s ~ Dirichlet(ones(nmixtures))

    y = datavar(Vector{Float64}, n)

    means = tuple(m...)
    precs = tuple(w...)

    for i in 1:n
        z[i] ~ Categorical(s)
        y[i] ~ NormalMixture(z[i], means, precs)
    end

    return s, z, m, w, y
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function inference_univariate(data, n_its, constraints, options)
    n = length(data)
    model, (s, m1, w1, m2, w2, z, y) = univariate_gaussian_mixture_model(constraints, options, n)

    mswitch = keep(Marginal)
    mm1 = keep(Marginal)
    mm2 = keep(Marginal)
    mw1 = keep(Marginal)
    mw2 = keep(Marginal)

    fe = keep(Float64)

    m1sub = subscribe!(getmarginal(m1), mm1)
    m2sub = subscribe!(getmarginal(m2), mm2)
    w1sub = subscribe!(getmarginal(w1), mw1)
    w2sub = subscribe!(getmarginal(w2), mw2)
    switchsub = subscribe!(getmarginal(s), mswitch)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    setmarginal!(s, vague(Beta))
    setmarginal!(m1, NormalMeanVariance(-2.0, 1e3))
    setmarginal!(m2, NormalMeanVariance(2.0, 1e3))
    setmarginal!(w1, vague(GammaShapeRate))
    setmarginal!(w2, vague(GammaShapeRate))

    for i in 1:n_its
        update!(y, data)
    end

    unsubscribe!((fesub, switchsub, m1sub, m2sub, w1sub, w2sub))

    return mswitch, mm1, mm2, mw1, mw2, fe
end

function inference_multivariate(rng, L, nmixtures, data, viters, constraints, options)
    n = length(data)

    model, (s, z, m, w, y) = multivariate_gaussian_mixture_model(constraints, options, rng, L, nmixtures, n)

    means_estimates  = keep(Vector{Marginal})
    precs_estimates  = keep(Vector{Marginal})
    switch_estimates = keep(Marginal)
    fe_values        = keep(Float64)

    s_sub = subscribe!(getmarginal(s), switch_estimates)
    m_sub = subscribe!(getmarginals(m), means_estimates)
    p_sub = subscribe!(getmarginals(w), precs_estimates)
    f_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe_values)

    setmarginal!(s, vague(Dirichlet, nmixtures))

    basis_v = [1.0, 0.0]

    for i in 1:nmixtures
        # Assume we now only approximate location of cluters's mean
        approximate_angle_prior = ((2π + +rand(rng)) / nmixtures) * (i - 1)
        approximate_basis_v = L / 2 * (basis_v .+ rand(rng, 2))
        approximate_rotation = [
            cos(approximate_angle_prior) -sin(approximate_angle_prior)
            sin(approximate_angle_prior) cos(approximate_angle_prior)
        ]
        mean_mean_prior = approximate_rotation * approximate_basis_v
        mean_mean_cov = [1e6 0.0; 0.0 1e6]

        setmarginal!(m[i], MvNormalMeanCovariance(mean_mean_prior, mean_mean_cov))
        setmarginal!(w[i], Wishart(3, [1e2 0.0; 0.0 1e2]))
    end

    for i in 1:viters
        update!(y, data)
    end

    unsubscribe!((s_sub, m_sub, p_sub, f_sub))

    return switch_estimates, means_estimates, precs_estimates, fe_values
end

@testset "Gaussian Mixture Model" begin
    @testset "Univariate" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        n = 150

        rng = StableRNG(12345)

        switch = [1 / 3, 2 / 3]
        z      = rand(rng, Categorical(switch), n)
        y      = Vector{Float64}(undef, n)

        μ1 = -10.0
        μ2 = 10.0
        w  = 1.777

        dists = [
            Normal(μ1, sqrt(inv(w))),
            Normal(μ2, sqrt(inv(w)))
        ]

        for i in 1:n
            y[i] = rand(rng, dists[z[i]])
        end
        ## -------------------------------------------- ##
        ## Inference execution
        constraints = @constraints begin
            q(z, s, m1, m2, w1, w2) = q(z)q(s)q(m1)q(w1)q(m2)q(w2)
        end

        # Execute inference for different constraints and option specification
        results = map(
            (specs) -> inference_univariate(y, 10, specs[1], specs[2]),
            [
                (DefaultConstraints, model_options(default_factorisation = MeanField())),
                (constraints, model_options())
            ]
        )

        fresult = results[begin]

        # All execution must be equivalent (check against first)
        foreach(results[begin+1:end]) do result
            foreach(zip(fresult, result)) do (l, r)
                @test ReactiveMP.getvalues(l) == ReactiveMP.getvalues(r)
            end
        end

        mswitch, mm1, mm2, mw1, mw2, fe = fresult
        ## -------------------------------------------- ##
        # Test inference results
        @test length(mswitch) === 10
        @test length(mm1) === 10
        @test length(mm2) === 10
        @test length(mw1) === 10
        @test length(mw2) === 10
        @test length(fe) === 10 && all(filter(e -> abs(e) > 1e-3, diff(getvalues(fe))) .< 0)
        @test abs(last(fe) - 280.3276104) < 0.01

        ms = mean(last(mswitch))

        @test ((abs(ms - switch[1]) < 0.1) || (abs(ms - switch[2]) < 0.1))

        ems = sort([last(mm1), last(mm2)], by = mean)
        rms = sort([μ1, μ2])

        foreach(zip(rms, ems)) do (r, e)
            @test abs(r - mean(e)) < 0.2
        end

        ews = sort([last(mw1), last(mw2)], by = mean)
        rws = sort([w, w])

        foreach(zip(rws, ews)) do (r, e)
            @test abs(r - mean(e)) < 0.2
        end
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "univariate_gmm_model_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "univariate_gmm_model_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        dim(d) = (a) -> map(r -> r[d], a)
        mp = plot(mean.(mm1), ribbon = var.(mm1) .|> sqrt, label = "m1 prediction")
        mp = plot!(mean.(mm2), ribbon = var.(mm2) .|> sqrt, label = "m2 prediction")
        mp = plot!(mp, [μ1], seriestype = :hline, label = "real m1")
        mp = plot!(mp, [μ2], seriestype = :hline, label = "real m2")

        wp =
            plot(mean.(mw1), ribbon = var.(mw1) .|> sqrt, label = "w1 prediction", legend = :bottomleft, ylim = (-1, 3))
        wp = plot!(wp, [w], seriestype = :hline, label = "real w1")
        wp = plot!(wp, mean.(mw2), ribbon = var.(mw2) .|> sqrt, label = "w2 prediction")
        wp = plot!(wp, [w], seriestype = :hline, label = "real w2")

        swp = plot(mean.(mswitch), ribbon = var.(mswitch) .|> sqrt, label = "Switch prediction")

        swp = plot!(swp, [switch[1]], seriestype = :hline, label = "switch[1]")
        swp = plot!(swp, [switch[2]], seriestype = :hline, label = "switch[2]")

        fep = plot(fe[2:end], label = "Free Energy", legend = :bottomleft)

        p = plot(mp, wp, swp, fep, layout = @layout([a b; c d]), size = (1000, 700))
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark inference_univariate(
                $y,
                10,
                $DefaultConstraints,
                model_options(default_factorisation = MeanField())
            ) seconds = 15
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end

    @testset "Multivariate" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        rng = StableRNG(43)

        L         = 50.0
        nmixtures = 3
        n_samples = 500

        probvec = ones(nmixtures)
        probvec = probvec ./ sum(probvec)

        switch = Categorical(probvec)

        gaussians = map(1:nmixtures) do index
            angle      = 2π / nmixtures * (index - 1)
            basis_v    = L * [1.0, 0.0]
            rotationm  = [cos(angle) -sin(angle); sin(angle) cos(angle)]
            mean       = rotationm * basis_v
            covariance = Matrix(Hermitian(rotationm * [10.0 0.0; 0.0 20.0] * transpose(rotationm)))
            return MvNormal(mean, covariance)
        end

        z = rand(rng, switch, n_samples)
        y = Vector{Vector{Float64}}(undef, n_samples)

        for i in 1:n_samples
            y[i] = rand(rng, gaussians[z[i]])
        end
        ## -------------------------------------------- ##
        ## Inference execution

        constraints = @constraints begin
            q(z, s, m, w) = q(z)q(s)q(m)q(w)
            q(m) = q(m[begin]) .. q(m[end]) # Mean-field over `m`
            q(w) = q(w[begin]) .. q(w[end]) # Mean-field over `w`
        end

        # Execute inference for different constraints and option specification
        results = map(
            (specs) -> inference_multivariate(specs[1], L, nmixtures, y, 25, specs[2], specs[3]),
            [
                (StableRNG(42), DefaultConstraints, model_options(default_factorisation = MeanField())),
                (StableRNG(42), constraints, model_options())
            ]
        )

        fresult = results[begin]

        # All execution must be equivalent (check against first)
        foreach(results[begin+1:end]) do result
            foreach(zip(fresult, result)) do (l, r)
                @test ReactiveMP.getvalues(l) == ReactiveMP.getvalues(r)
            end
        end

        s, m, w, fe = fresult
        ## -------------------------------------------- ##
        # Test inference results
        @test length(s) === 25
        @test length(m) === 25
        @test length(w) === 25
        @test length(fe) === 25
        @test all(filter(e -> abs(e) > 1e-3, diff(getvalues(fe))) .< 0)
        @test abs(last(fe) - 3442.4015524445967) < 0.01

        ems = sort(mean.(last(m)), by = x -> atan(x[2] / x[1]))
        rms = sort(mean.(gaussians), by = x -> atan(x[2] / x[1]))

        foreach(zip(ems, rms)) do (estimated, real)
            @test norm(normalize(estimated) .- normalize(real)) < 0.1
        end
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "multivariate_gmm_model_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "multivariate_gmm_model_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        sdim(n) = (a) -> map(d -> d[n], a)

        pe = plot(xlim = (-1.5L, 1.5L), ylim = (-1.5L, 1.5L))

        rp = scatter(y |> sdim(1), y |> sdim(2))

        pe = scatter!(pe, y |> sdim(1), y |> sdim(2))

        e_means = mean.(m[end])
        e_precs = mean.(w[end])

        for (e_m, e_w) in zip(e_means, e_precs)
            gaussian = MvNormal(e_m, Matrix(Hermitian(inv(e_w))))
            pe = contour!(
                pe,
                range(-2L, 2L, step = 0.25),
                range(-2L, 2L, step = 0.25),
                (x, y) -> pdf(gaussian, [x, y]),
                levels = 7,
                colorbar = false
            )
        end

        pfe = plot(fe[2:end], label = "Free Energy")

        p = plot(rp, pe, pfe, size = (600, 600), layout = @layout([a b; c]))

        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark inference_multivariate(
                StableRNG(123),
                $L,
                $nmixtures,
                $y,
                25,
                $DefaultConstraints,
                model_options(default_factorisation = MeanField())
            ) seconds = 15
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end
end

end
