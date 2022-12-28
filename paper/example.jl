# This file contains an example code for the `paper.md` for Journal of Open Source Software

using RxInfer, StableRNGs, Plots

Δt = 0.01 # -- Time step resolution
G = 9.81   # -- Gravitational acceleration

# Nonlinear state-transition function `f`
# See Example 3.7 in the "Bayesian Filtering and Smoothing" Simo Sarkka
f(x) = [x[1] + x[2] * Δt, x[2] - G * sin(x[1]) * Δt]

@model function pendulum()
    # Define reactive inputs for the `prior` 
    # of the current angle state
    prior_mean = datavar(Vector{Float64})
    prior_cov  = datavar(Matrix{Float64})

    previous_state ~ MvNormal(μ = prior_mean, Σ = prior_cov)
    # Use `f` as state transition function
    state ~ f(previous_state)

    # Assign a prior for the noise component
    noise_shape = datavar(Float64)
    noise_scale = datavar(Float64)
    noise ~ Gamma(noise_shape, noise_scale)

    # Define reactive input for the `observation`
    observation = datavar(Float64)
    observation ~ Normal(μ = dot([1.0, 0.0], state), τ = noise)
end

@constraints function pendulum_constraint()
    # Assume the `state` and the `noise` are independent
    q(state, noise) = q(state)q(noise)
end

@meta function pendulum_meta()
    # Use the `Linearization` approximation method 
    # around the (potentially) non-linear function `g`
    f() -> Linearization()
end

# Generate dummy data
function dataset(; T, precision = 1.0, seed = 42)
    rng = StableRNG(seed)

    # Initialize data array
    timesteps = (1:T) .* Δt
    states = zeros(2, T)
    observations = zeros(T)

    # Initial states
    states[:, 1] = [0.5, 0.0]

    for t in 2:T
        # State transition
        states[:, t] = f(states[:, t - 1])

        # Observation likelihood, we observe only the first component
        observations[t] = rand(rng, NormalMeanPrecision(states[1, t][1], precision))
    end

    # We return `states` only for testing purposes
    return timesteps, states, observations
end

function experiment(observations)

    # The `@autoupdates` structure defines how to update 
    # the priors for the next observation
    autoupdates = @autoupdates begin
        # Update `prior` automatically as soon as 
        # we have a new posterior for the `state`
        prior_mean  = mean(q(state))
        prior_cov   = cov(q(state))
        noise_shape = shape(q(noise))
        noise_scale = scale(q(noise))
    end

    results = rxinference(
        model = pendulum(),
        constraints = pendulum_constraint(),
        meta = pendulum_meta(),
        autoupdates = autoupdates,
        data = (observation = observations,),
        initmarginals = (
            # We assume a relatively good prior for the very first state
            state = MvNormalMeanPrecision([0.5, 0.0], [100.0 0.0; 0.0 100.0]),
            # And we assign a vague prior for the noise component
            noise = Gamma(1.0, 100.0)
        ),
        # We indicate that we want to keep a history of estimated 
        # states and the noise component
        historyvars = (state = KeepLast(), noise = KeepLast()),
        keephistory = length(observations),
        # We perform 5 VMP iterations on each observation
        iterations = 5,
        # We start the inference procedure automatically
        autostart = true
    )

    return results
end

@info "Generating dataset..."

timesteps, states, observations = dataset(T = 1000, precision = 1.0)

@info "Running inference..."

results = experiment(observations)

inferred_states = results.history[:state]

# Use the `pgf` backend for the Plots, requires LaTeX installation
# As well as the `PGFPlotsX.jl` package installed
# Comment this line if you don't have needed libraries installed
pgfplotsx();

@info "Generating plots..."

p = plot(timesteps, states[1, :], ylim = (-5, 2.5), label = "Real signal", xlabel = "Time (in s)", ylabel = "Pendulum angle (in radians)")
p = scatter!(p, timesteps, observations, ms = 2, alpha = 0.5, label = "Noisy observations")
p = plot!(p, timesteps, getindex.(mean.(inferred_states), 1), ribbon = getindex.(std.(inferred_states), 1, 1), label = "Inferred states")

lensrange = 170:250
lensx = [timesteps[lensrange[begin]], timesteps[lensrange[end]]]
lensy = [minimum(states[1, lensrange]) - 0.15, maximum(states[1, lensrange]) + 0.15]

p = lens!(p, lensx, lensy, inset = (1, bbox(0.1, 0.5, 0.4, 0.4)))

display(p)

@info "Saving the inference results plot..."

savefig(p, "inference.pdf")

# Benchmark, see the documentation for the `BenchmarkTools`
# for more information on the benchmarking syntax

@info "Running benchmarks..."

using BenchmarkTools

# See the optimization flags in the `compile.sh` file
benchmark = @benchmark experiment(data[3]) setup=(data = dataset(T = 1000; precision = 1.0)) seconds = 15

@info "Writing the benchmark results..."

open("benchmark.txt", "w") do file 
    show(file, "text/plain", benchmark)
    versioninfo(file, verbose = true)
end
