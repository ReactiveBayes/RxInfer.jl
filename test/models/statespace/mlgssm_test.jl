@testitem "Linear Gaussian State Space Model" begin
    using BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    ## Model definition #1
    @model function multivariate_lgssm_model(y, x0, A, B, Q, P)
        x_prior ~ MvNormal(μ = mean(x0), Σ = cov(x0))
        x_prev = x_prior

        for i in eachindex(y)
            x[i] ~ MvNormal(μ = A * x_prev, Σ = Q)
            y[i] ~ MvNormal(μ = B * x[i], Σ = P)
            x_prev = x[i]
        end
    end

    ## Model definition #2
    @model function state_transition(y_next, x_next, x_prev, A, B, P, Q)
        x_next ~ MvNormal(μ = A * x_prev, Σ = Q)
        y_next ~ MvNormal(μ = B * x_next, Σ = P)
    end

    @model function multivariate_lgssm_model_with_submodel(y, x0, A, B, Q, P)
        x_prev ~ MvNormal(μ = mean(x0), Σ = cov(x0))
        for i in eachindex(y)
            x[i] ~ state_transition(y_next = y[i], x_prev = x_prev, A = A, B = B, P = P, Q = Q)
            x_prev = x[i]
        end
    end

    ## Model definition #3
    @model function prod_distributions(a, b, c)
        a ~ b * c
    end

    @model function state_transition_with_submodel(y_next, x_next, x_prev, A, B, P, Q)
        x_next ~ MvNormal(μ = prod_distributions(b = A, c = x_prev), Σ = Q)
        y_next ~ MvNormal(μ = prod_distributions(b = B, c = x_next), Σ = P)
    end

    @model function multivariate_lgssm_model_with_several_submodel(y, x0, A, B, Q, P)
        x_prev ~ MvNormal(μ = mean(x0), Σ = cov(x0))
        for i in eachindex(y)
            x[i] ~ state_transition_with_submodel(y_next = y[i], x_prev = x_prev, A = A, B = B, P = P, Q = Q)
            x_prev = x[i]
        end
    end

    ## Inference definition
    function multivariate_lgssm_inference(model, data, x0, A, B, Q, P)
        return infer(model = model(x0 = x0, A = A, B = B, Q = Q, P = P), data = (y = data,), free_energy = true, options = (limit_stack_depth = 500,))
    end

    ## Data creation
    function generate_data(rng, A, B, Q, P)
        x_prev = [10.0, -10.0]

        x = Vector{Vector{Float64}}(undef, n)
        y = Vector{Vector{Float64}}(undef, n)

        for i in 1:n
            x[i] = rand(rng, MvNormal(A * x_prev, Q))
            y[i] = rand(rng, MvNormal(B * x[i], P))
            x_prev = x[i]
        end

        return x, y
    end

    # Seed for reproducibility
    rng = StableRNG(1234)
    # We will model 2-dimensional observations with rotation matrix `A`
    # To avoid clutter we also assume that matrices `A`, `B`, `P` and `Q`
    # are known and fixed for all time-steps
    θ = π / 35
    A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    B = diageye(2)
    Q = diageye(2)
    P = 25.0 .* diageye(2)
    n = 1_000 # Number of observations
    x, y = generate_data(rng, A, B, Q, P)
    x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))
    ## -------------------------------------------- ##
    ## Inference execution
    v_results = []
    v_marginals = []
    v_fe = []

    for model in [multivariate_lgssm_model, multivariate_lgssm_model_with_submodel, multivariate_lgssm_model_with_several_submodel]
        mresult = multivariate_lgssm_inference(model, y, x0, A, B, Q, P)
        xmarginals = mresult.posteriors[:x]
        fe = mresult.free_energy
        push!(v_results, mresult)
        push!(v_marginals, xmarginals)
        push!(v_fe, fe)
    end

    for (result, xmarginals, fe) in zip(v_results, v_marginals, v_fe)
        ## Test inference results
        @test length(xmarginals) === n
        # We use 3.0var instead of 3.0std here for easier dot broadcasting with mean
        @test all((mean.(xmarginals) .- 3.0 .* var.(xmarginals)) .< x .< (mean.(xmarginals) .+ 3.0 .* var.(xmarginals)))
        @test all(isposdef.(cov.(xmarginals)))
        @test length(fe) === 1
        @test abs(last(fe) - 6275.9015944677) < 0.01
    end

    # Check that all results are equal (check against the first one)
    for (xmarginals, fe) in zip(v_marginals, v_fe)
        @test all(v -> v[1] == v[2], zip(xmarginals, first(v_marginals)))
        @test all(v -> v[1] == v[2], zip(fe, first(v_fe)))
    end

    ## Create output plots
    @test_plot "models" "mlgssm" begin
        mresult = first(v_results)
        xmarginals = first(v_marginals)
        fe = first(v_fe)

        slicedim(dim) = (a) -> map(e -> e[dim], a)

        subrange = 100:500
        px = plot()

        px = plot!(px, x[subrange] |> slicedim(1), label = "Hidden Signal (dim-1)", color = :orange)
        px = plot!(px, x[subrange] |> slicedim(2), label = "Hidden Signal (dim-2)", color = :green)

        px = plot!(
            px,
            mean.(xmarginals)[subrange] |> slicedim(1),
            ribbon = var.(xmarginals)[subrange] |> slicedim(1) .|> sqrt,
            fillalpha = 0.5,
            label = "Estimated Signal (dim-1)",
            color = :teal
        )
        px = plot!(
            px,
            mean.(xmarginals)[subrange] |> slicedim(2),
            ribbon = var.(xmarginals)[subrange] |> slicedim(2) .|> sqrt,
            fillalpha = 0.5,
            label = "Estimated Signal (dim-1)",
            color = :violet
        )

        return px
    end

    @test_benchmark "models" "mlgssm" multivariate_lgssm_inference($multivariate_lgssm_model, $y, $x0, $A, $B, $Q, $P)
end
