module RxInferModelsAutoregressiveTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

## Model definition
## -------------------------------------------- ##
@model [default_factorisation = MeanField()] function ar_model(n, order)
    x = datavar(Vector{Float64}, n)
    y = datavar(Float64, n)

    γ ~ GammaShapeRate(1.0, 1.0)
    θ ~ MvNormalMeanPrecision(zeros(order), diageye(order))

    for i in 1:n
        y[i] ~ NormalMeanPrecision(dot(x[i], θ), γ)
    end

    return x, y, θ, γ
end

@model [default_factorisation = MeanField()] function lar_model(T::Type{Multivariate}, n, order, c, stype, τ)

    # Parameter priors
    γ ~ GammaShapeRate(1.0, 1.0)
    θ ~ MvNormalMeanPrecision(zeros(order), diageye(order))

    # We create a sequence of random variables for hidden states
    x = randomvar(n)
    # As well a sequence of observartions
    y = datavar(Float64, n)

    ct = constvar(c)
    # We assume observation noise to be known
    cτ = constvar(τ)

    # Prior for first state
    x0 ~ MvNormalMeanPrecision(zeros(order), diageye(order))

    x_prev = x0

    # AR process requires extra meta information
    meta = ARMeta(Multivariate, order, stype)

    for i in 1:n
        # Autoregressive node uses structured factorisation assumption between states
        x[i] ~ AR(x_prev, θ, γ) where {q = q(y, x)q(γ)q(θ), meta = meta}
        y[i] ~ NormalMeanPrecision(dot(ct, x[i]), cτ)
        x_prev = x[i]
    end

    return x, y, θ, γ
end
@model [default_factorisation = MeanField()] function lar_model(T::Type{Univariate}, n, order, c, stype, τ)

    # Parameter priors
    γ ~ GammaShapeRate(1.0, 1.0)
    θ ~ NormalMeanPrecision(0.0, 1.0)

    # We create a sequence of random variables for hidden states
    x = randomvar(n)
    # As well a sequence of observartions
    y = datavar(Float64, n)

    ct = constvar(c)
    # We assume observation noise to be known
    cτ = constvar(τ)

    # Prior for first state
    x0 ~ NormalMeanPrecision(0.0, 1.0)

    x_prev = x0

    # AR process requires extra meta information
    meta = ARMeta(Univariate, order, stype)

    for i in 1:n
        x[i] ~ AR(x_prev, θ, γ) where {q = q(y, x)q(γ)q(θ), meta = meta}
        y[i] ~ NormalMeanPrecision(ct * x[i], cτ)
        x_prev = x[i]
    end

    return x, y, θ, γ
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function ar_inference(inputs, outputs, order, niter)
    n = length(outputs)

    model, (x, y, θ, γ) = ar_model(model_options(limit_stack_depth = 500), n, order)

    γ_buffer = keep(Marginal)
    θ_buffer = keep(Marginal)
    fe       = keep(Float64)

    γ_sub = subscribe!(getmarginal(γ), γ_buffer)
    θ_sub = subscribe!(getmarginal(θ), θ_buffer)
    f_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    setmarginal!(γ, GammaShapeRate(1.0, 1.0))

    for _ in 1:niter
        update!(x, inputs)
        update!(y, outputs)
    end

    unsubscribe!((γ_sub, θ_sub, f_sub))

    return γ_buffer, θ_buffer, fe
end

function lar_init_marginals!(::Type{Multivariate}, order, γ, θ)
    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order), diageye(order)))
end

function lar_init_marginals!(::Type{Univariate}, order, γ, θ)
    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, NormalMeanPrecision(0.0, 1.0))
end

function lar_inference(data, order, artype, stype, niter, τ)

    # We build a full graph based on nber of observatios
    n = length(data)

    # Depending on the order of AR process `c` is
    # either a nber or a vector
    c = ReactiveMP.ar_unit(artype, order)

    # Note that to run inference for huge model it might be necessary to pass extra
    # `model_options(limit_stack_depth = 100)` as a first argument to limit stack depth during recursive inference procedure
    model, (x, y, θ, γ) = lar_model(artype, n, order, c, stype, τ)

    # We are going to keep `γ` and `θ` estimates for all VMP iterations
    # But `buffer` only last posterior estimates for a sequence of hidden states `x`
    # We also will keep Bethe Free Energy in `fe`
    γ_buffer = keep(Marginal)
    θ_buffer = keep(Marginal)
    x_buffer = buffer(Marginal, n)
    fe       = keep(Float64)

    γsub  = subscribe!(getmarginal(γ), γ_buffer)
    θsub  = subscribe!(getmarginal(θ), θ_buffer)
    xsub  = subscribe!(getmarginals(x), x_buffer)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    lar_init_marginals!(artype, order, γ, θ)

    # We update data several times to perform several VMP iterations
    for i in 1:niter
        update!(y, data)
    end

    # It is important to unsubscribe from running observables
    unsubscribe!((γsub, θsub, xsub, fesub))

    return γ_buffer, θ_buffer, x_buffer, fe
end

@testset "Autoregressive model" begin
    @testset "Autoregressive model" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        function ar_ssm(series, order)
            inputs = [reverse!(series[1:order])]
            outputs = [series[order+1]]
            for x in series[order+2:end]
                push!(inputs, vcat(outputs[end], inputs[end])[1:end-1])
                push!(outputs, x)
            end
            return inputs, outputs
        end
        rng = StableRNG(1234)
        series = randn(rng, 1_000)
        ## -------------------------------------------- ##
        ## Inference execution and test inference results
        for order in 1:5
            inputs, outputs = ar_ssm(series, order)
            γ_buffer, θ_buffer, fe = ar_inference(inputs, outputs, order, 15)
            @test length(γ_buffer) === 15
            @test length(θ_buffer) === 15
            @test length(fe) === 15
            @test last(fe) < first(fe)
            @test all(filter(e -> abs(e) > 1e-3, diff(getvalues(fe))) .< 0)
        end
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        benchmark_output = joinpath(base_output, "ar_model_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            inputs5, outputs5 = ar_ssm(series, 5)
            benchmark = @benchmark ar_inference($inputs5, $outputs5, 5, 15) seconds = 15#
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end

    @testset "Latent autoregressive model" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        # The following coefficients correspond to stable poles
        coefs_ar_5 =
            [0.10699399235785655, -0.5237303489793305, 0.3068897071844715, -0.17232255282458891, 0.13323964347539288]

        function generate_lar_data(rng, n, θ, γ, τ)
            order        = length(θ)
            states       = Vector{Vector{Float64}}(undef, n + 3order)
            observations = Vector{Float64}(undef, n + 3order)

            γ_std = sqrt(inv(γ))
            τ_std = sqrt(inv(γ))

            states[1] = randn(rng, order)

            for i in 2:(n+3order)
                states[i]       = vcat(rand(rng, Normal(dot(θ, states[i-1]), γ_std)), states[i-1][1:end-1])
                observations[i] = rand(rng, Normal(states[i][1], τ_std))
            end

            return states[1+3order:end], observations[1+3order:end]
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
        ## -------------------------------------------- ##
        ## Inference execution

        # AR order 1
        γ, θ, xs, fe = lar_inference(observations, 1, Univariate, ARsafe(), 15, real_τ)
        @test length(xs) === n
        @test length(γ) === 15
        @test length(θ) === 15
        @test length(fe) === 15
        @test abs(last(fe) - 518.9182342) < 0.01
        @test all(filter(e -> abs(e) > 1e-3, diff(getvalues(fe))) .< 0)

        for i in 1:4
            γ, θ, xs, fe = lar_inference(observations, i, Multivariate, ARsafe(), 15, real_τ)
            @test length(xs) === n
            @test length(γ) === 15
            @test length(θ) === 15
            @test length(fe) === 15
        end

        # AR order 5
        γ, θ, xs, fe = lar_inference(observations, length(real_θ), Multivariate, ARsafe(), 15, real_τ)

        ## -------------------------------------------- ##
        ## Test inference results
        @test length(xs) === n
        @test length(γ) === 15
        @test length(θ) === 15
        @test length(fe) === 15
        @test abs(last(fe) - 514.66086) < 0.01
        @test all(filter(e -> abs(e) > 1e-1, diff(getvalues(fe))) .< 0)
        @test (mean(last(γ)) - 3.0std(last(γ)) < real_γ < mean(last(γ)) + 3.0std(last(γ)))
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "lar_model_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "lar_model_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        p1 = plot(first.(states), label = "Hidden state")
        p1 = scatter!(p1, observations, label = "Observations")
        p1 = plot!(
            p1,
            first.(mean.(xs)),
            ribbon = sqrt.(first.(var.(xs))),
            label = "Inferred states",
            legend = :bottomright
        )

        p2 = plot(mean.(γ), ribbon = std.(γ), label = "Inferred transition precision", legend = :bottomright)
        p2 = plot!([real_γ], seriestype = :hline, label = "Real transition precision")

        p3 = plot(getvalues(fe), label = "Bethe Free Energy")

        p = plot(p1, p2, p3, layout = @layout([a; b c]))
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark =
                @benchmark lar_inference(
                    $observations, length($real_θ), Multivariate, ARsafe(), 15, $real_τ
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
