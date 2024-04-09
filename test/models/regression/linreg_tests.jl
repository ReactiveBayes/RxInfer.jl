@testitem "Linear regression" begin
    using BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

    # Please use StableRNGs for random number generators

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    ## Model definition
    @model function linear_regression(x, y)
        a ~ Normal(mean = 0.0, var = 1.0)
        b ~ Normal(mean = 0.0, var = 1.0)
        for (i, k) in zip(eachindex(x), eachindex(y))
            y[k] ~ Normal(mean = x[i] * b + a, var = 1.0)
        end
    end

    @model function linear_regression_broadcasted(x, y)
        a ~ Normal(mean = 0.0, var = 1.0)
        b ~ Normal(mean = 0.0, var = 1.0)
        # Variance over-complicated for a purpose of checking that this expressions is allowed, 
        # it should be equal to `1.0`, but it shouldn't create any extra factor nodes
        y .~ Normal(mean = x .* b .+ a, var = det((diageye(2) .+ diageye(2)) ./ 2))
    end

    init = @initialization begin
        μ(b) = NormalMeanVariance(0.0, 100.0)
    end

    ## Inference definition
    function linreg_inference(modelfn, niters, xdata, ydata)
        return infer(model = modelfn(), data = (x = xdata, y = ydata), returnvars = (a = KeepLast(), b = KeepLast()), init = init, free_energy = true, iterations = niters)
    end

    ## Data creation
    reala = 10.0
    realb = -10.0

    N = 100

    rng = StableRNG(1234)

    xdata = collect(1:N) .+ 1 * randn(rng, N)
    ydata = reala .+ realb .* xdata

    ## Inference execution
    result  = linreg_inference(linear_regression, 25, xdata, ydata)
    resultb = linreg_inference(linear_regression_broadcasted, 25, xdata, ydata)

    ares = result.posteriors[:a]
    bres = result.posteriors[:b]
    fres = result.free_energy

    aresb = resultb.posteriors[:a]
    bresb = resultb.posteriors[:b]
    fresb = resultb.free_energy

    ## Test inference results
    @test mean(ares) ≈ mean(aresb) && var(ares) ≈ var(aresb) # Broadcasting may change the order of computations, so slight 
    @test mean(bres) ≈ mean(bresb) && var(bres) ≈ var(bresb) # differences are allowed
    @test all(fres .≈ fresb)
    @test isapprox(mean(ares), reala, atol = 5)
    @test isapprox(mean(bres), realb, atol = 0.1)
    @test fres[end] < fres[2] # Loopy belief propagation has no guaranties though

    @test_benchmark "models" "linreg" linreg_inference(linear_regression, 25, $xdata, $ydata)
    @test_benchmark "models" "linreg_broadcasted" linreg_inference(linear_regression_broadcasted, 25, $xdata, $ydata)
end
