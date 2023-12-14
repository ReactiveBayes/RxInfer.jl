@testitem "Hidden Markov Model" begin
    using BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    ## Model definition
    @model function hidden_markov_model(n)
        A ~ MatrixDirichlet(ones(3, 3))
        B ~ MatrixDirichlet([10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0])

        s_0 ~ Categorical(fill(1.0 / 3.0, 3))

        s = randomvar(n)
        x = datavar(Vector{Float64}, n)

        s_prev = s_0

        for t in 1:n
            s[t] ~ Transition(s_prev, A)
            x[t] ~ Transition(s[t], B)
            s_prev = s[t]
        end
    end

    @constraints function hidden_markov_constraints()
        q(s, s_0, A, B) = q(s, s_0)q(A)q(B)
    end

    ## Inference definition
    function hidden_markov_model_inference(data, vmp_iters)
        return infer(
            model = hidden_markov_model(length(data)),
            constraints = hidden_markov_constraints(),
            data = (x = data,),
            options = (limit_stack_depth = 500,),
            free_energy = true,
            initmarginals = (A = vague(MatrixDirichlet, 3, 3), B = vague(MatrixDirichlet, 3, 3), s = vague(Categorical, 3)),
            iterations = vmp_iters,
            returnvars = (s = KeepEach(), A = KeepEach(), B = KeepEach())
        )
    end

    ## Data creation
    function rand_vec(rng, distribution::Categorical)
        k = ncategories(distribution)
        s = zeros(k)
        s[rand(rng, distribution)] = 1.0
        s
    end

    function generate_data(rng, n_samples)

        # Transition probabilities (some transitions are impossible)
        A = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9]
        # Observation noise
        B = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9]
        # Initial state
        s_0 = [1.0, 0.0, 0.0]
        # Generate some data
        s = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the states
        x = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the observations

        s_prev = s_0

        for t in 1:n_samples
            a = A * s_prev
            s[t] = rand_vec(rng, Categorical(a ./ sum(a)))
            b = B * s[t]
            x[t] = rand_vec(rng, Categorical(b ./ sum(b)))
            s_prev = s[t]
        end

        return x, s
    end

    rng = StableRNG(123)
    x_data, s_data = generate_data(rng, 100)

    ## Inference execution
    result = hidden_markov_model_inference(x_data, 20)

    sbuffer = result.posteriors[:s]
    Abuffer = result.posteriors[:A]
    Bbuffer = result.posteriors[:B]
    fe = result.free_energy

    ## Test inference results
    @test length(sbuffer) === 20 && all(b -> length(b) === 100, sbuffer)
    @test length(Abuffer) === 20
    @test length(Bbuffer) === 20
    @test length(fe) === 20 && all(filter(e -> abs(e) > 1e-3, diff(fe)) .< 0)
    @test abs(last(fe) - 60.614480654) < 0.01

    ## Create output plots
    @test_plot "models" "hmm" begin
        p1 = scatter(argmax.(s_data))
        p2 = scatter(argmax.(ReactiveMP.probvec.(sbuffer[end])))
        p3 = plot(fe)
        p = plot(p1, p2, p3, layout = @layout([a b; c]))
        return p
    end

    @test_benchmark "models" "mlgssm" hidden_markov_model_inference($x_data, 20)
end
