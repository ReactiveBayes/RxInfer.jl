@testitem "Latent autoregressive model" begin
    using StableRNGs, Plots, BenchmarkTools

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    # The latent autoregressive model may run in different regimes, e.g. 
    # - it can be a Univariate autoregressive process where we only need to infer a single AR coefficient
    # - it can be a Multivariate autoregressive process where we need to infer a vector of AR coefficients
    #   - but the vector can be of length 1, which is essentially a Univariate AR process
    # The test below defines different combinations of the model and checks that the inference works correctly.

    # The prior on `θ` in case if the Autoregressive process is Multivariate
    @model function lar_multivariate_θ_prior(θ, order)
        θ ~ MvNormal(mean = zeros(order), precision = diageye(order))
    end

    # The prior on `θ` in case if the Autoregressive process is Univariate
    @model function lar_univariate_θ_prior(θ, order)
        θ ~ Normal(mean = 0.0, precision = 1.0)
    end

    # The prior on `x0` in case if the Autoregressive process is Multivariate
    @model function lar_multivariate_x0_prior(x0, order)
        x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))
    end

    # The prior on `x0` in case if the Autoregressive process is Univariate
    @model function lar_univariate_x0_prior(x0, order)
        x0 ~ Normal(mean = 0.0, precision = 1.0)
    end

    # The state transition with the `AR` node for the Multivariate case
    @model function lar_multivariate_state_transition(order, stype, observation, x_next, x_prev, θ, γ, c, τ)
        x_next ~ AR(x_prev, θ, γ) where {meta = ARMeta(Multivariate, order, stype)}
        observation ~ Normal(mean = dot(c, x_next), precision = τ)
    end

    # The state transition with the `AR` node for the Univariate case
    @model function lar_univariate_state_transition(order, stype, observation, x_next, x_prev, θ, γ, c, τ)
        x_next ~ AR(x_prev, θ, γ) where {meta = ARMeta(Univariate, order, stype)}
        observation ~ Normal(mean = c * x_next, precision = τ)
    end

    # Generic LAR model implementation, which depends on `θprior`, `x0prior`, `state_transition` submodels
    @model function lar_model(θprior, x0prior, state_transition, y, c, τ, order, stype)
        γ ~ Gamma(shape = 1.0, rate = 1.0)
        θ ~ θprior(order = order)
        x0 ~ x0prior(order = order)
        x_prev = x0
        for i in eachindex(y)
            x[i] ~ state_transition(observation = y[i], x_prev = x_prev, θ = θ, γ = γ, c = c, τ = τ, order = order, stype = stype)
            x_prev = x[i]
        end
    end

    # We specify structured constraints over the states and mean-field over the autoregressive coefficients/parameters
    @constraints function lar_constraints()
        q(x, x0, γ, θ) = q(x, x0)q(γ)q(θ)
    end

    # The initial marginals for the autoregressive coefficients/parameters depending on the type of the AR process
    lar_init_marginals(::Type{Multivariate}, order) = (γ = GammaShapeRate(1.0, 1.0), θ = MvNormalMeanPrecision(zeros(order), diageye(order)))
    lar_init_marginals(::Type{Univariate}, order) = (γ = GammaShapeRate(1.0, 1.0), θ = NormalMeanPrecision(0.0, 1.0))

    # The model constructor depending on the type of the AR process
    lar_make_model(::Type{Multivariate}, c, τ, stype, order) = lar_model(
        θprior = lar_multivariate_θ_prior, x0prior = lar_multivariate_x0_prior, state_transition = lar_multivariate_state_transition, order = order, c = c, stype = stype, τ = τ
    )
    lar_make_model(::Type{Univariate}, c, τ, stype, order) = lar_model(
        θprior = lar_univariate_θ_prior, x0prior = lar_univariate_x0_prior, state_transition = lar_univariate_state_transition, order = order, c = c, stype = stype, τ = τ
    )

    function lar_inference(data, order, artype, stype, τ, iterations)
        c = ReactiveMP.ar_unit(artype, order)
        return infer(
            model = lar_make_model(artype, c, τ, stype, order),
            data = (y = data,),
            initmarginals = lar_init_marginals(artype, order),
            returnvars = (γ = KeepEach(), θ = KeepEach(), x = KeepLast()),
            constraints = lar_constraints(),
            iterations = iterations,
            free_energy = Float64
        )
    end

    # The following coefficients correspond to stable poles
    coefs_ar_5 = [0.10699399235785655, -0.5237303489793305, 0.3068897071844715, -0.17232255282458891, 0.13323964347539288]

    function generate_lar_data(rng, n, θ, γ, τ)
        order        = length(θ)
        states       = Vector{Vector{Float64}}(undef, n + 3order)
        observations = Vector{Float64}(undef, n + 3order)

        γ_std = sqrt(inv(γ))
        τ_std = sqrt(inv(γ))

        states[1] = randn(rng, order)

        for i in 2:(n + 3order)
            states[i]       = vcat(rand(rng, Normal(dot(θ, states[i - 1]), γ_std)), states[i - 1][1:(end - 1)])
            observations[i] = rand(rng, Normal(states[i][1], τ_std))
        end

        return states[(1 + 3order):end], observations[(1 + 3order):end]
    end

    # Seed for reproducibility
    rng = StableRNG(123)

    # Number of observations in synthetic dataset
    n = 500

    # AR process parameters
    real_γ = 5.0
    real_τ = 5.0
    real_θ = coefs_ar_5
    states, observations = generate_lar_data(rng, n, real_θ, real_γ, real_τ)

    # Test AR(1) + Univariate
    result = lar_inference(observations, 1, Univariate, ARsafe(), real_τ, 15)
    qs     = result.posteriors
    fe     = result.free_energy

    (γ, θ, xs) = (qs[:γ], qs[:θ], qs[:x])

    @test length(xs) === n
    @test length(γ) === 15
    @test length(θ) === 15
    @test length(fe) === 15
    @test abs(last(fe) - 518.9182342) < 0.01
    @test last(fe) < first(fe)
    @test all(filter(e -> abs(e) > 1e-3, diff(fe)) .< 0)

    # Test AR(k) + Multivariate
    for k in 1:4
        local result = lar_inference(observations, k, Multivariate, ARsafe(), real_τ, 15)
        local qs = result.posteriors
        local fe = result.free_energy
        local (γ, θ, xs) = (qs[:γ], qs[:θ], qs[:x])

        @test length(xs) === n
        @test length(γ) === 15
        @test length(θ) === 15
        @test length(fe) === 15
        @test last(fe) < first(fe)
    end

    # AR(5) + Multivariate
    result = lar_inference(observations, length(real_θ), Multivariate, ARsafe(), real_τ, 15)
    qs     = result.posteriors
    fe     = result.free_energy

    (γ, θ, xs) = (qs[:γ], qs[:θ], qs[:x])

    @test length(xs) === n
    @test length(γ) === 15
    @test length(θ) === 15
    @test length(fe) === 15
    @test abs(last(fe) - 514.66086) < 0.01
    @test all(filter(e -> abs(e) > 1e-1, diff(fe)) .< 0)
    @test (mean(last(γ)) - 3.0std(last(γ)) < real_γ < mean(last(γ)) + 3.0std(last(γ)))

    @test_plot "models" "lar" begin
        p1 = plot(first.(states), label = "Hidden state")
        p1 = scatter!(p1, observations, label = "Observations")
        p1 = plot!(p1, first.(mean.(xs)), ribbon = sqrt.(first.(var.(xs))), label = "Inferred states", legend = :bottomright)

        p2 = plot(mean.(γ), ribbon = std.(γ), label = "Inferred transition precision", legend = :bottomright)
        p2 = plot!([real_γ], seriestype = :hline, label = "Real transition precision")

        p3 = plot(fe, label = "Bethe Free Energy")

        p = plot(p1, p2, p3, layout = @layout([a; b c]))
    end

    @test_benchmark "models" "lar" lar_inference($observations, length($real_θ), Multivariate, ARsafe(), 15, $real_τ)
end
