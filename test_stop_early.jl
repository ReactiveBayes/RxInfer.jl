using RxInfer
using StableRNGs
import ExponentialFamily: softmax 

function generate_multinomial_data(rng=StableRNG(123);N = 20, k=9, nsamples = 1000)
    Ψ = randn(rng, k)
    p = softmax(Ψ)
    X = rand(rng, Multinomial(N, p), nsamples)
    X= [X[:,i] for i in 1:size(X,2)];
    return X, Ψ,p
end

nsamples = 5000
N = 30
k = 40
X, Ψ, p = generate_multinomial_data(N=N, k=k, nsamples=nsamples);

@model function multinomial_model(obs, N, ξ_ψ, W_ψ)
    ψ ~ MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
    obs .~ MultinomialPolya(N, ψ) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ))}
end

result = infer(
    model = multinomial_model(ξ_ψ=zeros(k-1), W_ψ=rand(Wishart(3, diageye(k-1))), N=N),
    data = (obs=X, ),
    iterations = 50,
    free_energy = true,
    showprogress = true,
    options = (
        limit_stack_depth = 100,
    ),
    callbacks=(
        after_iteration=StopEarlyIterationStrategy(0.01),
    )
)

