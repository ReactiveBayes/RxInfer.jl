@testitem "Multinomial regression with MultinomialPolya node" begin
    using BenchmarkTools, Plots, Distributions, LinearAlgebra, StableRNGs, ExponentialFamily.LogExpFunctions
    function generate_multinomial_data(rng=StableRNG(123); N = 3, k=3, nsamples = 5000)
        ψ = randn(rng, k)
        p = ReactiveMP.softmax(ψ)
    
        X = rand(rng, Multinomial(N, p), nsamples)
        X = [X[:,i] for i in axes(X,2)]
        return X, ψ,p
    end
    
    N = 50
    k = 40

    X, ψ,p = generate_multinomial_data(;N=N, k=k)
    
    @model function multinomial_model(y, N, ξ_ψ, W_ψ)
        ψ ~ MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
        for i in eachindex(y)
            y[i] ~ MultinomialPolya(N, ψ) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ))}
        end
    end

    
    result = infer(
        model = multinomial_model(ξ_ψ=zeros(k-1), W_ψ=rand(Wishart(k, diageye(k-1))), N=N),
        data = (y=X,),
        iterations = 50,
        free_energy = false,
        showprogress = false,
        returnvars = KeepLast(),
        options = (
            limit_stack_depth = 100, 
        )
    )

    function logistic_stic_breaking(m)
        Km1 = length(m)

        p = Array{Float64}(undef, Km1+1)
        p[1] = logistic(m[1])
        for i in 2:Km1
            p[i] = logistic(m[i])*(1 - sum(p[1:i-1]))
        end
        p[end] = 1 - sum(p[1:end-1])
        return p
    end

    m = mean(result.posteriors[:ψ])
    
    pest = logistic_stic_breaking(m)

    mse = mean((pest - p).^2)
    @test mse < 1e-5
end


@testitem "Multinomial regression - online inference" begin
    using BenchmarkTools, Plots, Distributions, LinearAlgebra, StableRNGs, ExponentialFamily.LogExpFunctions
    function generate_multinomial_data(rng=StableRNG(123); N = 3, k=3, nsamples = 500000)
        ψ = randn(rng, k)
        p = ReactiveMP.softmax(ψ)
    
        X = rand(rng, Multinomial(N, p), nsamples)
        X = [X[:,i] for i in axes(X,2)]
        return X, ψ,p
    end
    N = 50
    k = 40
    X, ψ,p = generate_multinomial_data(;N=N, k=k)
    
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
        # iterations = 1,
        initialization = init,
        autoupdates = auto(),
        keephistory = length(X),
        free_energy = false,
        showprogress = false,
    )

    m = result.history[:ψ][end]

    function logistic_stic_breaking(m)
        Km1 = length(m)

        p = Array{Float64}(undef, Km1+1)
        p[1] = logistic(m[1])
        for i in 2:Km1
            p[i] = logistic(m[i])*(1 - sum(p[1:i-1]))
        end
        p[end] = 1 - sum(p[1:end-1])
        return p
    end
    pest = logistic_stic_breaking(mean(m))

    mse = mean((pest - p).^2)
    @test mse < 1e-4

end