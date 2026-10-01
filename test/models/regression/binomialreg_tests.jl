@testitem "Linear regression with BinomialPolya node" begin
    using BenchmarkTools, Plots, Dates, LinearAlgebra, StableRNGs
    using PolyaMessagePassingRules

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    function generate_synthetic_binomial_data(
        n_samples::Int, true_beta::Vector{Float64}; seed::Int = 42
    )
        rng = StableRNG(seed)
        n_features = length(true_beta)
        # Generate design matrix X
        X = randn(rng, n_samples, n_features)

        # Generate number of trials for each observation
        n_trials = rand(rng, 5:20, n_samples)

        # Compute logits and probabilities
        logits = X * true_beta
        probs = 1 ./ (1 .+ exp.(-logits))

        # Generate binomial outcomes
        y = [rand(rng, Binomial(n_trials[i], probs[i])) for i in 1:n_samples]

        return X, y, n_trials
    end
    n_samples = 1000
    n_features = 2
    true_beta = [-1.0, 0.6]
    n_iterations = 100
    n_sims = 20

    @model function binomial_model(prior_xi, prior_precision, n_trials, X, y)
        β ~ MvNormalWeightedMeanPrecision(prior_xi, prior_precision)
        for i in eachindex(y)
            y[i] ~ BinomialPolya(X[i], n_trials[i], β)
        end
    end

    # The node reads the message on `β`'s own edge, which starts from the prior
    @initialization function binomial_initialization(prior_xi, prior_precision)
        μ(β) = MvNormalWeightedMeanPrecision(prior_xi, prior_precision)
    end

    function binomial_inference(
        binomial_model, iterations, X, y, n_trials, n_features
    )
        return infer(
            model = binomial_model(
                prior_xi = zeros(n_features),
                prior_precision = diageye(n_features),
            ),
            data = (X = X, y = y, n_trials = n_trials),
            initialization = binomial_initialization(
                zeros(n_features), diageye(n_features)
            ),
            iterations = iterations,
            free_energy = true,
            options = (limit_stack_depth = 100,),
        )
    end

    function run_simulation(
        n_sims::Int,
        n_samples::Int,
        true_beta::Vector{Float64};
        iterations = n_iterations,
    )
        # Storage for results
        n_features = length(true_beta)
        coverage = Vector{Vector{Float64}}(undef, n_sims)
        fes = Vector{Vector{Float64}}(undef, n_sims)
        for sim in 1:n_sims
            # Generate new dataset
            X, y, n_trials = generate_synthetic_binomial_data(
                n_samples, true_beta, seed = sim
            )
            X = [collect(row) for row in eachrow(X)]

            # Run inference
            results = binomial_inference(
                binomial_model, iterations, X, y, n_trials, n_features
            )
            # Extract posterior parameters
            post = results.posteriors[:β][end]
            m = mean(post)
            v = var(post)
            estimates = map((x, y) -> Normal(x, sqrt(y)), m, v)
            coverage[sim] = map((d, b) -> cdf(d, b), estimates, true_beta)
            fes[sim] = results.free_energy
        end

        return coverage, fes
    end

    function in_credible_interval(x, lwr = 0.025, upr = 0.975)
        return x >= lwr && x <= upr
    end

    coverage, fes = run_simulation(n_sims, n_samples, true_beta)
    for i in 1:n_sims
        @test fes[i][end] < fes[i][1]
    end
    coverages = Vector{Float64}(undef, n_features)
    for i in 1:n_features
        coverages[i] =
            sum(in_credible_interval.(getindex.(coverage, i))) / n_sims
        @test coverages[i] >= 0.8
    end

    @test_benchmark "models" "binomialreg" binomial_inference(
        binomial_model, $n_iterations, $X, $y, $n_trials
    )
end

@testitem "BinomialPolya draws from the rng the context option gives" begin
    using StableRNGs, LinearAlgebra
    using PolyaMessagePassingRules

    # `samples = k` averages over draws from `ctx.rng`, which `options = (context = (rng = …,),)`
    # supplies: two seeds give different posteriors, one seed the same twice. Without sampling the
    # rules use means, and the seed changes nothing.
    @model function binomial_model(X, n, y, algorithm)
        β ~ MvNormalWeightedMeanPrecision(zeros(2), diageye(2))
        for i in eachindex(y)
            y[i] ~ BinomialPolya(X[i], n[i], β) where {algorithm = algorithm}
        end
    end

    rng = StableRNG(42)
    X = [randn(rng, 2) for _ in 1:30]
    n = fill(10, 30)
    y = [rand(rng, Binomial(10, 1 / (1 + exp(x[2] - x[1])))) for x in X]
    initialization = @initialization(
        begin
            μ(β) = MvNormalWeightedMeanPrecision(zeros(2), diageye(2))
        end
    )
    posterior(algorithm, seed) = mean(
        infer(
            model = binomial_model(algorithm = algorithm),
            data = (X = X, n = n, y = y),
            initialization = initialization,
            iterations = 5,
            options = (context = (rng = StableRNG(seed),),),
        ).posteriors[:β][end],
    )

    sampling = BinomialPolyaApproximation(samples = 50)
    @test posterior(sampling, 1) != posterior(sampling, 2)
    @test posterior(sampling, 1) == posterior(sampling, 1)
    @test posterior(BinomialPolyaApproximation(), 1) ==
        posterior(BinomialPolyaApproximation(), 2)
end
