# A precompile workload: a few small models of the kinds most people start with, so that their
# first inference runs native code cached when RxInfer was precompiled rather than compiling the
# engine, GraphPPL and the rules again. Every call adds to RxInfer's precompile time, so the
# workload is kept to the common paths; `free_energy = true` is left out, since what it compiles
# depends on the model. Disable it with PrecompileTools' preference:
# `PrecompileTools.Preferences.set_preferences!(RxInfer, "precompile_workload" => false; force = true)`.
module PrecompileWorkload

using ..RxInfer
using PrecompileTools: @setup_workload, @compile_workload

# belief propagation along a chain
@model function state_space(y)
    x_prev ~ Normal(mean = 0.0, variance = 100.0)
    for i in eachindex(y)
        x[i] ~ Normal(mean = x_prev, variance = 1.0)
        y[i] ~ Normal(mean = x[i], variance = 1.0)
        x_prev = x[i]
    end
end

# variational message passing under a mean-field factorisation
@model function iid(y)
    μ ~ Normal(mean = 0.0, variance = 100.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    for i in eachindex(y)
        y[i] ~ Normal(mean = μ, precision = τ)
    end
end

# a conjugate pair, exact
@model function beta_bernoulli(y)
    θ ~ Beta(1.0, 1.0)
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

@setup_workload begin
    y = [0.1, -0.2, 0.3, 0.05, -0.1]
    coin = [1.0, 0.0, 1.0]
    initialization = @initialization(q(τ) = GammaShapeRate(1.0, 1.0))
    @compile_workload begin
        infer(model = state_space(), data = (y = y,), session = nothing)
        infer(
            model = iid(),
            data = (y = y,),
            constraints = MeanField(),
            initialization = initialization,
            iterations = 2,
            session = nothing,
        )
        infer(model = beta_bernoulli(), data = (y = coin,), session = nothing)
        # the scheduler that long chains need changes the type of every stream; a limit this low
        # makes the short chain reach it, so the path that continues on a new task compiles too
        infer(
            model = state_space(),
            data = (y = y,),
            options = (limit_stack_depth = 2,),
            session = nothing,
        )
    end
end

end
