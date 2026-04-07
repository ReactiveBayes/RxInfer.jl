using RxInfer
using Random, Distributions
using BenchmarkTools, ColorSchemes, Colors, Plots, StableRNGs

# We create model using GraphPPL.jl package interface with @model macro
# For simplicity of the example we consider all matrices to be known and constant
@model function linear_gaussian_ssm_smoothing(y, A, B, P, Q)
    
    # Set a prior distribution for x[1]
    x[1] ~ MvNormal(μ = [ 0.0, 0.0 ], Σ = [ 100.0 0.0; 0.0 100.0 ]) 
    y[1] ~ MvNormal(μ = B * x[1], Σ = Q)
    
    for t in 2:length(y)
        x[t] ~ MvNormal(μ = A * x[t - 1], Σ = P)
        y[t] ~ MvNormal(μ = B * x[t], Σ = Q)    
    end

end

# It is also possible to create a single time step of the graph
# for filtering algorithm
@model function linear_gaussian_ssm_filtering(x_min_t_mean, x_min_t_cov, y_t, A, B, P, Q)    
    x_min_t ~ MvNormal(μ = x_min_t_mean, Σ = x_min_t_cov)
    x_t     ~ MvNormal(μ = A * x_min_t, Σ = P)
    y_t     ~ MvNormal(μ = B * x_t, Σ = Q)
end

function generate_data(n, A, B, P, Q; seed = 42)
    Random.seed!(seed)

    x_prev = zeros(2)
    x      = Vector{Vector{Float64}}(undef, n)
    y      = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        x[i]   = rand(MvNormal(A * x_prev, P))
        y[i]   = rand(MvNormal(B * x[i], Q))
        x_prev = x[i]
    end
   
    return x, y
end

n = 200
θ = π / 15
A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
B = [ 1.3 0.0; 0.0 0.7 ]
P = [ 0.05 0.0; 0.0 0.05 ]
Q = [ 10.0 0.0; 0.0 10.0 ]


real_x, real_y = generate_data(n, A, B, P, Q);

# Inference procedure for full graph
function rxinfer_inference_smoothing(observations, A, B, P, Q)
    n = length(observations) 
    
    result = infer(
        model = linear_gaussian_ssm_smoothing(A = A, B = B, P = P, Q = Q),
        data = (y = observations, ),
        options = (limit_stack_depth = 500, )
    )
    
    return result.posteriors[:x]
end

# Inference procedure for single time step graph and filtering
function rxinfer_inference_filtering(observations, A, B, P, Q)
    n = length(observations) 
    
    autoupdates = @autoupdates begin 
        x_min_t_mean, x_min_t_cov = mean_cov(q(x_t))
    end
    
    result = infer(
        model = linear_gaussian_ssm_filtering(A = A, B = B, P = P, Q = Q),
        data = (y_t = observations, ),
        autoupdates = autoupdates,
        initialization = @initialization(q(x_t) = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 100.0 0.0; 0.0 100.0 ])),
        historyvars = (x_t = KeepLast(), ),
        keephistory = n
    )
    
    return result.history[:x_t]
end

real_x_1000, real_y_1000 = generate_data(1000, A, B, P, Q);

result = @benchmark rxinfer_inference_filtering($real_y_1000, $A, $B, $P, $Q)

show(result)
