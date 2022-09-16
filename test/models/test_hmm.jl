module ReactiveMPModelsHMMTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

## Model definition
## -------------------------------------------- ##
@model function transition_model(n)
    A ~ MatrixDirichlet(ones(3, 3))
    B ~ MatrixDirichlet([10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0])

    s_0 ~ Categorical(fill(1.0 / 3.0, 3))

    s = randomvar(n)
    x = datavar(Vector{Float64}, n)

    s_prev = s_0

    for t in 1:n
        s[t] ~ Transition(s_prev, A) where { q = q(out, in)q(a) }
        x[t] ~ Transition(s[t], B)
        s_prev = s[t]
    end

    return s, x, A, B
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function inference(data, vmp_iters)
    n = length(data)

    # TODO: add meanfield

    model, (s, x, A, B) = transition_model(model_options(limit_stack_depth = 500), n)

    sbuffer = keep(Vector{Marginal})
    Abuffer = keep(Marginal)
    Bbuffer = keep(Marginal)
    fe      = ScoreActor(Float64)

    ssub  = subscribe!(getmarginals(s), sbuffer)
    Asub  = subscribe!(getmarginal(A), Abuffer)
    Bsub  = subscribe!(getmarginal(B), Bbuffer)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    setmarginal!(A, vague(MatrixDirichlet, 3, 3))
    setmarginal!(B, vague(MatrixDirichlet, 3, 3))

    foreach(s) do svar
        setmarginal!(svar, vague(Categorical, 3))
    end

    for i in 1:vmp_iters
        update!(x, data)
    end

    unsubscribe!(ssub)
    unsubscribe!(Asub)
    unsubscribe!(Bsub)
    unsubscribe!(fesub)

    return map(getvalues, (sbuffer, Abuffer, Bbuffer, fe))
end

@testset "Hidden Markov Model" begin
    @testset "Full graph" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
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
        ## -------------------------------------------- ##
        ## Inference execution
        sbuffer, Abuffer, Bbuffer, fe = inference(x_data, 20)
        ## -------------------------------------------- ##
        ## Test inference results
        @test length(sbuffer) === 20
        @test length(Abuffer) === 20
        @test length(Bbuffer) === 20
        @test length(fe) === 20 && all(filter(e -> abs(e) > 1e-3, diff(fe)) .< 0)
        @test abs(last(fe) - 60.614480654) < 0.01

        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "hmm_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "hmm_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots        
        p1 = scatter(argmax.(s_data))
        p2 = scatter(argmax.(ReactiveMP.probvec.(sbuffer[end])))
        p3 = plot(fe)
        p = plot(p1, p2, p3, layout = @layout([a b; c]))
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark inference($x_data, 20) seconds = 15
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end
end

end
