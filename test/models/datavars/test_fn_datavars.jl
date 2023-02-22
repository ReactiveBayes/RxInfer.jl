module RxInferModelsDatavarsTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# Please use StableRNGs for random number generators

include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

## Model definition
@model function sum_datavars_as_gaussian_mean_1()
    a = datavar(Float64)
    b = datavar(Float64)
    y = datavar(Float64)

    x ~ Normal(mean = a + b, variance = 1.0)
    y ~ Normal(mean = x, variance = 1.0)
end

@model function sum_datavars_as_gaussian_mean_2()
    a = datavar(Float64)
    b = datavar(Float64)
    c = constvar(0.0) # Should not change the result
    y = datavar(Float64)

    x ~ Normal(mean = (a + b) + c, variance = 1.0)
    y ~ Normal(mean = x, variance = 1.0)
end

@model function ratio_datavars_as_gaussian_mean()
    a = datavar(Float64)
    b = datavar(Float64)
    y = datavar(Float64)

    x ~ Normal(mean = a / b, variance = 1.0)
    y ~ Normal(mean = x, variance = 1.0)
end

@model function idx_datavars_as_gaussian_mean()
    a = datavar(Vector{Float64})
    b = datavar(Matrix{Float64})
    y = datavar(Float64)

    x ~ Normal(mean = dot(a[1:2], b[1:2,1]), variance = 1.0)
    y ~ Normal(mean = x, variance = 1.0)
end

# Inference function
function fn_datavars_inference(modelfn, adata, bdata, ydata)
    return inference(model = modelfn(), data = (a = adata, b = bdata, y = ydata), free_energy = true)
end

@testset "datavars" begin
    adata = 2.0
    bdata = 1.0
    ydata = 0.0

    # Test inference in model with linear operation on datavars
    result = fn_datavars_inference(sum_datavars_as_gaussian_mean_1, adata, bdata, ydata)
    model = result.model
    xres = result.posteriors[:x]
    fres = result.free_energy

    isproxy_vars = [ReactiveMP.isproxy(var) for var in model.variables.data]
    @test length(isproxy_vars) == 4
    @test sum(isproxy_vars) == 1
    @test typeof(xres) <: NormalDistributionsFamily
    @test isapprox(mean(xres), 1.5, atol = 0.1)
    @test isapprox(fres[end], 3.51551, atol = 0.1)

    result = fn_datavars_inference(sum_datavars_as_gaussian_mean_2, adata, bdata, ydata)
    model = result.model
    xres = result.posteriors[:x]
    fres = result.free_energy

    isproxy_vars = [ReactiveMP.isproxy(var) for var in model.variables.data]
    @test length(isproxy_vars) == 5
    @test sum(isproxy_vars) == 2
    @test typeof(xres) <: NormalDistributionsFamily
    @test isapprox(mean(xres), 1.5, atol = 0.1)
    @test isapprox(fres[end], 3.51551, atol = 0.1)

    # Test inference in model with nonlinear operation on datavars
    result = fn_datavars_inference(ratio_datavars_as_gaussian_mean, adata, bdata, ydata)
    model = result.model
    xres = result.posteriors[:x]
    fres = result.free_energy

    isproxy_vars = [ReactiveMP.isproxy(var) for var in model.variables.data]
    @test length(isproxy_vars) == 4
    @test sum(isproxy_vars) == 1
    @test typeof(xres) <: NormalDistributionsFamily
    @test isapprox(mean(xres), 1.0, atol = 0.1)
    @test isapprox(fres[end], 2.26551, atol = 0.1)

    # Test with indexing datavariables
    A_data = [1.0, 2.0, 3.0]
    B_data = [1.0 0.5; 0.5 1.0]
    @test_broken result = fn_datavars_inference(idx_datavars_as_gaussian_mean, A_data, B_data, ydata)
end

end
