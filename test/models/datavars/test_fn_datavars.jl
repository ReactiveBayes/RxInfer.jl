module RxInferModelsDatavarsTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# Please use StableRNGs for random number generators

include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

## Model definition
@model function sum_datavars_as_gaussian_mean()

    a = datavar(Float64)
    b = datavar(Float64)
    y = datavar(Float64)

    x ~ Normal(mean = a+b, variance = 1.0)
    y ~ Normal(mean = x, variance = 1.0)
end

@model function ratio_datavars_as_gaussian_mean()

    a = datavar(Float64)
    b = datavar(Float64)
    y = datavar(Float64)

    x ~ Normal(mean = a/b, variance = 1.0)
    y ~ Normal(mean = x, variance = 1.0)
end

# Inference function
function fn_datavars_inference(modelfn, adata, bdata, ydata)
    return inference(
        model = modelfn(),
        data = (a = adata, b = bdata, y = ydata),
        free_energy = true
    )
end

@testset "datavars" begin

    # Test model construction
    model,_ = create_model(sum_datavars_as_gaussian_mean())
    isproxy_vars = [ReactiveMP.isproxy(var) for var in model.variables.data]
    @test length(isproxy_vars) == 4
    @test sum(isproxy_vars) == 1
    @test ReactiveMP.isproxy(model.variables.data[isproxy_vars])

    adata = 2.0
    bdata = 1.0
    ydata = 0.0

    # Test inference in model with linear operation on datavars
    result = fn_datavars_inference(sum_datavars_as_gaussian_mean, adata, bdata, ydata)
    xres = result.posteriors[:x]
    fres = result.free_energy

    @test typeof(xres) <: NormalDistributionsFamily
    @test isapprox(mean(xres), 1.5, atol = 0.1)
    @test isapprox(fres[end], 3.51551, atol = 0.1)

    # Test inference in model with linear operation on datavars
    result = fn_datavars_inference(ratio_datavars_as_gaussian_mean, adata, bdata, ydata)
    xres = result.posteriors[:x]
    fres = result.free_energy

    @test typeof(xres) <: NormalDistributionsFamily
    @test isapprox(mean(xres), 1.0, atol = 0.1)
    @test isapprox(fres[end], 2.26551, atol = 0.1)

end

end