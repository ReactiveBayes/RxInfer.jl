module RxInferModelsGMMTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, LinearAlgebra, StableRNGs

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

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

        m[i] ~ MvNormal(mean = mean_mean_prior, cov = mean_mean_cov)
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
end

function inference_multivariate(rng, L, nmixtures, data, viters, constraints)
    basis_v = [1.0, 0.0]

    minitmarginals = []
    winitmarginals = []

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

        push!(minitmarginals, MvNormalMeanCovariance(mean_mean_prior, mean_mean_cov))
        push!(winitmarginals, Wishart(3, [1e2 0.0; 0.0 1e2]))
    end

    return infer(
        model = multivariate_gaussian_mixture_model(rng, L, nmixtures, length(data)),
        data = (y = data,),
        constraints = constraints,
        returnvars = KeepEach(),
        free_energy = Float64,
        iterations = viters,
        initmarginals = (s = vague(Dirichlet, nmixtures), m = minitmarginals, w = winitmarginals)
    )
end

@testset "Multivariate Gaussian Mixture model" begin
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

    ## Inference execution
    constraints = @constraints begin
        q(z, s, m, w) = q(z)q(s)q(m)q(w)
        q(m) = q(m[begin]) .. q(m[end]) # Mean-field over `m`
        q(w) = q(w[begin]) .. q(w[end]) # Mean-field over `w`
    end

    # Execute inference for different constraints and option specification
    results = map((specs) -> inference_multivariate(specs[1], L, nmixtures, y, 25, specs[2]), [(StableRNG(42), MeanField()), (StableRNG(42), constraints)])

    fresult = results[begin]

    # All execution must be equivalent (check against first)
    foreach(results[(begin + 1):end]) do result
        foreach(zip(fresult.posteriors, result.posteriors)) do (l, r)
            @test l == r
        end
    end

    s = fresult.posteriors[:s]
    m = fresult.posteriors[:m]
    w = fresult.posteriors[:w]
    fe = fresult.free_energy
    ## -------------------------------------------- ##
    # Test inference results
    @test length(s) === 25
    @test length(m) === 25
    @test length(w) === 25
    @test length(fe) === 25
    @test all(filter(e -> abs(e) > 1e-3, diff(fe)) .< 0)
    @test abs(last(fe) - 3442.4015524445967) < 0.01

    ems = sort(mean.(last(m)), by = x -> atan(x[2] / x[1]))
    rms = sort(mean.(gaussians), by = x -> atan(x[2] / x[1]))

    foreach(zip(ems, rms)) do (estimated, real)
        @test norm(normalize(estimated) .- normalize(real)) < 0.1
    end

    @test_plot "models" "gmm_multivariate" begin
        sdim(n) = (a) -> map(d -> d[n], a)

        pe = plot(xlim = (-1.5L, 1.5L), ylim = (-1.5L, 1.5L))

        rp = scatter(y |> sdim(1), y |> sdim(2))

        pe = scatter!(pe, y |> sdim(1), y |> sdim(2))

        e_means = mean.(m[end])
        e_precs = mean.(w[end])

        for (e_m, e_w) in zip(e_means, e_precs)
            gaussian = MvNormal(e_m, Matrix(Hermitian(inv(e_w))))
            pe = contour!(pe, range(-2L, 2L, step = 0.25), range(-2L, 2L, step = 0.25), (x, y) -> pdf(gaussian, [x, y]), levels = 7, colorbar = false)
        end

        pfe = plot(fe[2:end], label = "Free Energy")

        return plot(rp, pe, pfe, size = (600, 600), layout = @layout([a b; c]))
    end

    @test_benchmark "models" "gmm_multivariate" inference_multivariate(StableRNG(123), $L, $nmixtures, $y, 25, MeanField())
end

end
