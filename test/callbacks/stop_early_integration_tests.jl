@testitem "StopEarlyIterationStrategy integration test on Gaussian state-space model" begin
    using RxInfer, Distributions, LinearAlgebra, StableRNGs

    @model function gaussian_state_space_model(y, x0, process_var, obs_var)
        x_prev ~ Normal(mean = mean(x0), var = var(x0))

        for i in eachindex(y)
            x[i] ~ Normal(mean = x_prev, var = process_var)
            y[i] ~ Normal(mean = x[i], var = obs_var)
            x_prev = x[i]
        end
    end

    function make_data(rng, n, process_var, obs_var)
        hidden = Vector{Float64}(undef, n)
        obs = Vector{Float64}(undef, n)

        hidden[1] = rand(rng, Normal(0.0, sqrt(process_var)))
        obs[1] = rand(rng, Normal(hidden[1], sqrt(obs_var)))

        for i in 2:n
            hidden[i] = rand(rng, Normal(hidden[i - 1], sqrt(process_var)))
            obs[i] = rand(rng, Normal(hidden[i], sqrt(obs_var)))
        end

        return obs
    end

    function run_inference(data, max_iterations, tol)
        initialization = @initialization begin
            q(x_prev) = NormalMeanVariance(0.0, 1.0)
            q(x) = NormalMeanVariance(0.0, 1.0)
        end

        return infer(
            model = gaussian_state_space_model(x0 = NormalMeanVariance(0.0, 10.0), process_var = 0.2, obs_var = 0.5),
            data = (y = data,),
            constraints = MeanField(),
            initialization = initialization,
            iterations = max_iterations,
            free_energy = true,
            showprogress = false,
            callbacks = (after_iteration = StopEarlyIterationStrategy(tol),)
        )
    end

    rng = StableRNG(123)
    max_iterations = 40
    data = make_data(rng, 200, 0.2, 0.5)

    result_loose = run_inference(data, max_iterations, 1e-2)
    result_strict = run_inference(data, max_iterations, 1e-12)

    loose_iters = length(result_loose.free_energy)
    strict_iters = length(result_strict.free_energy)

    @test loose_iters < max_iterations
    @test strict_iters <= max_iterations
    @test strict_iters >= loose_iters
end
