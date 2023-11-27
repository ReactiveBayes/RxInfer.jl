@testitem "Autoregressive model" begin
    using StableRNGs, BenchmarkTools
    
    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    @model function ar_model(n, order)
        x = datavar(Vector{Float64}, n)
        y = datavar(Float64, n)

        γ ~ Gamma(shape = 1.0, rate = 1.0)
        θ ~ MvNormal(mean = zeros(order), precision = diageye(order))

        for i in 1:n
            y[i] ~ Normal(mean = dot(x[i], θ), precision = γ)
        end
    end

    function ar_inference(inputs, outputs, order, niter)
        return inference(
            model         = ar_model(length(outputs), order),
            data          = (x = inputs, y = outputs),
            constraints   = MeanField(),
            options       = (limit_stack_depth = 500,),
            initmarginals = (γ = GammaShapeRate(1.0, 1.0),),
            returnvars    = (γ = KeepEach(), θ = KeepEach()),
            iterations    = niter,
            free_energy   = Float64
        )
    end

    function ar_ssm(series, order)
        inputs = [reverse!(series[1:order])]
        outputs = [series[order + 1]]
        for x in series[(order + 2):end]
            push!(inputs, vcat(outputs[end], inputs[end])[1:(end - 1)])
            push!(outputs, x)
        end
        return inputs, outputs
    end
    rng = StableRNG(1234)



    ## Inference execution and test inference results
    for order in 1:5
        series = randn(rng, 1_000)
        inputs, outputs = ar_ssm(series, order)
        result = ar_inference(inputs, outputs, order, 15)
        qs = result.posteriors

        (γ, θ) = (qs[:γ], qs[:θ])
        fe     = result.free_energy

        @test length(γ) === 15
        @test length(θ) === 15
        @test length(fe) === 15
        @test last(fe) < first(fe)
        @test all(filter(e -> abs(e) > 1e-3, diff(fe)) .< 0)
    end

    benchrng          = randn(StableRNG(32), 1_000)
    inputs5, outputs5 = ar_ssm(benchrng, 5)

    @test_benchmark "models" "ar" ar_inference($inputs5, $outputs5, 5, 15)
end