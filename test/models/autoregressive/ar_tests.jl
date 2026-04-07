@testitem "Autoregressive model" begin
    using StableRNGs, BenchmarkTools

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    function ar_ssm_data(series, order)
        inputs = [reverse!(series[1:order])]
        outputs = [series[order + 1]]
        for x in series[(order + 2):end]
            push!(inputs, vcat(outputs[end], inputs[end])[1:(end - 1)])
            push!(outputs, x)
        end
        return inputs, outputs
    end

    @model function ar_model(y, x, order)
        γ ~ Gamma(shape = 1.0, rate = 1.0)
        θ ~ MvNormal(mean = zeros(order), precision = diageye(order))
        # `i` and `k` should be the same here, but the code is more 
        # generic with `zip` over `eachindex`
        for (i, k) in zip(eachindex(y), eachindex(x))
            y[i] ~ Normal(mean = dot(x[k], θ), precision = γ)
        end
    end

    @constraints function ar_constraints()
        q(γ, θ) = q(γ)q(θ)
    end

    init = @initialization begin
        q(γ) = GammaShapeRate(1.0, 1.0)
    end

    function ar_inference(inputs, outputs, order, niter, constraints)
        return infer(
            model          = ar_model(order = order),
            data           = (x = inputs, y = outputs),
            constraints    = constraints,
            options        = (limit_stack_depth = 500,),
            initialization = init,
            returnvars     = (γ = KeepEach(), θ = KeepEach()),
            iterations     = niter,
            free_energy    = Float64
        )
    end

    rng = StableRNG(1234)

    ## Inference execution and test inference results
    for order in 1:5
        series = randn(rng, 1_000)
        inputs, outputs = ar_ssm_data(series, order)
        for constraints in [ar_constraints(), MeanField()]
            result = ar_inference(inputs, outputs, order, 15, constraints)

            qs = result.posteriors
            (γ, θ) = (qs[:γ], qs[:θ])
            fe = result.free_energy

            @test length(γ) === 15
            @test length(θ) === 15
            @test length(fe) === 15
            @test last(fe) < first(fe)
            @test all(filter(e -> abs(e) > 1e-3, diff(fe)) .< 0)
        end
    end

    benchrng          = randn(StableRNG(32), 1_000)
    inputs5, outputs5 = ar_ssm_data(benchrng, 5)

    @test_benchmark "models" "ar" ar_inference($inputs5, $outputs5, 5, 15)
end
