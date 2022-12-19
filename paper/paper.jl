# This file contains an example code for the `paper.md` for Journal of Open Source Software

using RxInfer, StableRNGs, Plots

Δt = 0.005 # -- Time step resolution
G = 9.81   # -- Gravitational acceleration

# Nonlinear state-transition function `f`
g(x) = [x[1] + x[2]*Δt, x[2] - G*sin(x[1])*Δt]

@model function pendulum()
    # Define reactive inputs for the `prior`
    prior_mean = datavar(Vector{Float64})
    prior_cov  = datavar(Matrix{Float64})

    # Define the `prior` state
    state_prior ~ MvNormal(μ = prior_mean, Σ = prior_cov)
    # Use `f` as state transition function
    state ~ g(state_prior)
    
    # Assign a prior for the noise component
    noise_shape = datavar(Float64)
    noise_scale = datavar(Float64)
    noise ~ Gamma(noise_shape, noise_scale)

    # Define reactive input for the `observation`
    observation = datavar(Float64)
    observation ~ Normal(μ = dot([ 1.0, 0.0 ], state), τ = noise)  
end

@constraints function pendulum_constraint()
    # Assume the `state` and the `noise`
    # are independent
    q(state, noise) = q(state)q(noise)
end

@meta function pendulum_meta()
    # Use the `Linearization` approximation method 
    # around the (potentially) non-linear function `f`
    g() -> Unscented()
end

# Generate dummy data
function dataset(; T, precision = 1.0, seed = 42)
    rng = StableRNG(seed)

    # Initialize data array
    timesteps = (1:T).*Δt
    states = zeros(2,T)
    observations = zeros(T,)

    # Initial states
    states[:, 1] = [ 0.99 , 0.0 ]

    for t = 2:T
        # State transition
        states[:, t] = g(states[:, t - 1])
        
        # Observation likelihood, we observe only the first component
        observations[t] = rand(rng, NormalMeanPrecision(states[1,t][1], precision))
    end

    # We return `states` only for testing purposes
    return timesteps, states, observations
end

timesteps, states, observations = dataset(T = 1000, precision = 0.5)

plot(timesteps, states[1, :])
scatter!(timesteps, observations, ms = 2, alpha = 0.5)

function experiment(observations)

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
        data = (observation = observations, ),
        initmarginals = (
            # We assume a relatively good prior for the very first state
            state = MvNormalMeanPrecision([ 0.99, 0.0 ], [ 100.0 0.0; 0.0 100.0 ]), 
            noise = Gamma(1.0, 100.0)
        ),
        historyvars = (state = KeepLast(), ),
        keephistory = length(observations),
        iterations = 10,
        autostart = true
    )

    return results
end

results = experiment(observations)

inferred_states = results.history[:state]

plot(timesteps, states[1, :], ylim = (-4, 1.5))
scatter!(timesteps, observations, ms = 2, alpha = 0.5)
plot!(timesteps, getindex.(mean.(inferred_states), 1), ribbon = getindex(cov.(inferred_states), 1, 1))

lensrange = 100:120
lensx = [ timesteps[lensrange[begin]], timesteps[lensrange[end]] ]
lensy = [ minimum(states[1, lensrange]), maximum(states[1, lensrange]) ]

lens!(lensx, lensy, inset = (1, bbox(0.1, 0.5, 0.4, 0.4)))

