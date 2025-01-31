@testitem "Multinomial regression with MultinomialPolya (offline inference) node" begin
    using BenchmarkTools, Plots, Distributions, LinearAlgebra, StableRNGs, ExponentialFamily.LogExpFunctions

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))
   
    N = 20
    k = 10
    nsamples = 5000 ###Needs 5000 for an accuracy of 1e-5
    X, ψ,p = generate_multinomial_data(;N=N, k=k, nsamples=nsamples)
    
    @model function multinomial_model(y, N, ξ_ψ, W_ψ)
        ψ ~ MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
        for i in eachindex(y)
            y[i] ~ MultinomialPolya(N, ψ) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ))}
        end
    end

    
    result = infer(
        model = multinomial_model(ξ_ψ=zeros(k-1), W_ψ=rand(Wishart(k, diageye(k-1))), N=N),
        data = (y=X,),
        iterations = 300,
        free_energy = true,
        showprogress = false,
        returnvars = KeepLast(),
        options = (
            limit_stack_depth = 100, 
        )
    )


    m = mean(result.posteriors[:ψ])  
    pest = logistic_stic_breaking(m)

    mse = mean((pest - p).^2)
    @test mse < 1e-5

    @test result.free_energy[end] < result.free_energy[1]
    @test result.free_energy[end] <= result.free_energy[end-1]
    @test abs(result.free_energy[end-1] - result.free_energy[end]) < 1e-12

end


@testitem "Multinomial regression - online inference" begin
    using BenchmarkTools, Plots, Distributions, LinearAlgebra, StableRNGs, ExponentialFamily.LogExpFunctions

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))
 
    N = 50
    k = 40
    nsamples = 5000
    X, ψ,p = generate_multinomial_data(;N=N, k=k, nsamples=nsamples)
    
    @model function multinomial_model(y, N, ξ_ψ, W_ψ, k)
        ψ ~ MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
        y ~ MultinomialPolya(N, ψ) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(zeros(k-1), diageye(k-1)))}
    end

    @autoupdates function auto()
        ξ_ψ, W_ψ = weightedmean_precision(q(ψ))
    end
    init = @initialization begin
        q(ψ) = MvNormalWeightedMeanPrecision(zeros(k-1), rand(Wishart(k, diageye(k-1))))
    end

    result = infer(
        model = multinomial_model(N = N, k = k),
        data = (y=X, ),
        initialization = init,
        iterations = 1,
        autoupdates = auto(),
        keephistory = length(X),
        free_energy = true,
        showprogress = false,
    )

    m = result.history[:ψ][end]

    pest = logistic_stic_breaking(mean(m))

    mse = mean((pest - p).^2)
    @test mse < 1e-3

    @test result.free_energy_final_only_history[end] < result.free_energy_final_only_history[1]
    #Free energy over time decreases in a noisy way. It is not a monotonic decrease. 

end

@testitem "Multinomial regression - prediction with missing data" begin
    using BenchmarkTools, Plots, Distributions, LinearAlgebra, StableRNGs, ExponentialFamily.LogExpFunctions

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))
 
    N = 5
    k = 4
    nsamples = 900
    X, ψ, p = generate_multinomial_data(;N=N, k=k, nsamples=nsamples)
    
    # Create dataset with missing values (every other sample)
    X_missing = Vector{Union{Vector{Int}, Missing}}(missing, nsamples)
    X_missing[1:600] = X[1:600]
    X_missing[600:end] .= missing

    @model function multinomial_model(y, N, ξ_ψ, W_ψ, k)
        ψ ~ MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
        for i in eachindex(y)
            y[i] ~ MultinomialPolya(N, ψ) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(zeros(k-1), diageye(k-1)))}
        end
    end

  
    result = infer(
        model = multinomial_model(N = N, k = k, ξ_ψ = zeros(k-1), W_ψ = rand(Wishart(k, diageye(k-1)))),
        data = (y=X_missing, ),
        iterations = 20,
        keephistory = length(X_missing),
        showprogress = false,
        options = (
            limit_stack_depth = 100, 
        )
    )

    # Check predictions for missing entries
    predicted_means = mean.(result.predictions[:y][end])
    actual_missing = [X[i] for i in 1:length(X_missing) if ismissing(X_missing[i])]
    
#     # Convert to probability estimates and calculate MSE
    pest_pred = [m ./ N for m in predicted_means]
    mse_pred = mean([sum((pred .- p).^2) for pred in pest_pred])
    @test mse_pred < 1e-3

end