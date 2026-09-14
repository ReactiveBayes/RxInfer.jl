@testitem "CompiledRunner: conjugate posteriors, constraints, and free energy" begin
    @model function compiled_coin(y)
        p ~ Beta(2.0, 3.0)
        for i in eachindex(y)
            y[i] ~ Bernoulli(p)
        end
    end
    @model function compiled_normal(y)
        x ~ NormalMeanVariance(0.0, 1.0)
        for i in eachindex(y)
            y[i] ~ NormalMeanVariance(x, 2.0)
        end
    end
    @model function compiled_mean_precision(y)
        x ~ NormalMeanVariance(0.0, 1.0)
        p ~ GammaShapeRate(2.0, 3.0)
        for i in eachindex(y)
            y[i] ~ NormalMeanPrecision(x, p)
        end
    end
    function compare(model, data; initialization = nothing, constraints = nothing, iterations = 100, free_energy = true)
        args = (; model, data, initialization, constraints, iterations, free_energy,
            returnvars = KeepLast(), session = nothing, disable_inference_error_hint = true)
        baseline = infer(; args...)
        for workers in unique((1, Threads.nthreads()))
            actual = infer(; args..., options = (runner = CompiledRunner(; workers),))
            @test actual.model.metadata[:execution_backend] === :compiled
            @test RxInfer.getmodel(actual.model).graph isa GraphPPL.CompactGraph
            for name in keys(baseline.posteriors)
                @test mean(actual.posteriors[name]) ≈ mean(baseline.posteriors[name]) rtol=1e-4 atol=1e-6
                @test var(actual.posteriors[name]) ≈ var(baseline.posteriors[name]) rtol=1e-4 atol=1e-6
            end
            if free_energy !== false
                @test actual.free_energy[end] ≈ baseline.free_energy[end] rtol=1e-8 atol=1e-8
            end
        end
    end
    compare(compiled_coin(), (; y = [1.0, 0.0, 1.0]); iterations = 1)
    compare(compiled_normal(), (; y = sin.(1:10)); iterations = 1, free_energy = Float32)
    init = @initialization begin
        q(x) = NormalMeanVariance(0.0, 1.0)
        q(p) = GammaShapeRate(2.0, 3.0)
    end
    constraints = @constraints begin
        q(x, p) = q(x)q(p)
    end
    compare(compiled_mean_precision(), (; y = sin.(1:10)); initialization = init, constraints)
    # A high-degree variable must not compile unused quadratic cavity products.
    args = (; model = compiled_normal(), data = (y = sin.(1:2048),), iterations = 1,
        returnvars = KeepLast(), session = nothing, disable_inference_error_hint = true)
    serial = infer(; args..., options = (runner = CompiledRunner(workers = 1),))
    parallel = infer(; args..., options = (runner = CompiledRunner(),))
    @test serial.posteriors == parallel.posteriors
    @test length(parallel.model.metadata[:inference_runner].operations) < 10 * 2048
    @test any(parallel.model.metadata[:inference_runner].phase_parallel)
end

@testitem "CompiledRunner: structured hidden Markov marginals" begin
    @model function compiled_hmm(y)
        A ~ DirichletCollection([8.0 2.0; 2.0 8.0])
        B ~ DirichletCollection([12.0 1.0; 1.0 12.0])
        s0 ~ Categorical([0.5, 0.5])
        previous = s0
        for i in eachindex(y)
            s[i] ~ DiscreteTransition(previous, A)
            y[i] ~ DiscreteTransition(s[i], B)
            previous = s[i]
        end
    end
    constraints = @constraints begin
        q(s, s0, A, B) = q(s, s0)q(A)q(B)
    end
    init = @initialization begin
        q(A) = DirichletCollection([8.0 2.0; 2.0 8.0])
        q(B) = DirichletCollection([12.0 1.0; 1.0 12.0])
        q(s) = Categorical([0.5, 0.5])
    end
    y = [i % 8 < 4 ? [1.0, 0.0] : [0.0, 1.0] for i in 1:24]
    args = (; model=compiled_hmm(), data=(; y), initialization=init, constraints,
        iterations=150, returnvars=KeepLast(), free_energy=true, session=nothing)
    baseline = infer(; args...)
    actual = infer(; args..., options=(runner=CompiledRunner(),))
    for name in (:A, :B, :s0), statistic in (mean, var)
        @test statistic(actual.posteriors[name]) ≈ statistic(baseline.posteriors[name]) rtol=1e-4 atol=1e-6
    end
    for statistic in (mean, var)
        @test statistic.(actual.posteriors[:s]) ≈ statistic.(baseline.posteriors[:s]) rtol=1e-4 atol=1e-6
    end
    @test actual.free_energy[end] ≈ baseline.free_energy[end] rtol=1e-4 atol=1e-6
end

@testitem "CompiledRunner: required self-messages and explicit rejection" begin
    @model function compiled_probit(y, dependencies)
        x[1] ~ NormalMeanVariance(0.0, 2.0)
        for i in eachindex(y)
            x[i+1] ~ NormalMeanVariance(x[i], 0.1)
            y[i] ~ Probit(x[i+1]) where {dependencies=dependencies}
        end
    end
    for policy in (nothing, RequireMessageFunctionalDependencies(in=NormalMeanVariance(0.0, 1.0)))
        args = (; model=compiled_probit(dependencies=policy), data=(y=[false, false, true, true, true],),
            iterations=100, returnvars=(x=KeepLast(),), free_energy=true, session=nothing)
        baseline = infer(; args...)
        actual = infer(; args..., options=(runner=CompiledRunner(),))
        @test mean.(actual.posteriors[:x]) ≈ mean.(baseline.posteriors[:x]) rtol=1e-4 atol=1e-6
        @test var.(actual.posteriors[:x]) ≈ var.(baseline.posteriors[:x]) rtol=1e-4 atol=1e-6
        @test actual.free_energy[end] ≈ baseline.free_energy[end] rtol=1e-4 atol=1e-6
    end
    @model function compiled_stream_policy(y)
        x ~ NormalMeanVariance(0.0, 1.0)
        y ~ NormalMeanVariance(x, 1.0) where {stream_postprocessors=identity}
    end
    @test_throws UnsupportedCompiledFeature infer(model=compiled_stream_policy(), data=(y=0.5,),
        options=(runner=CompiledRunner(),), session=nothing)
end

@testitem "CompiledRunner: gamma mixture and point-mass form constraints" begin
    @model function compiled_gamma_mixture(y)
        s ~ Dirichlet([2.0, 2.0])
        a[1] ~ GammaShapeRate(4.0, 2.0)
        a[2] ~ GammaShapeRate(8.0, 2.0)
        b[1] ~ GammaShapeRate(2.0, 2.0)
        b[2] ~ GammaShapeRate(2.0, 2.0)
        for i in eachindex(y)
            z[i] ~ Categorical(s)
            y[i] ~ GammaMixture(switch=z[i], a=a, b=b)
        end
    end
    constraints = @constraints begin
        q(s, a, b, z) = q(s)q(a)q(b)q(z)
        q(a) = q(a[begin]) .. q(a[end])
        q(b) = q(b[begin]) .. q(b[end])
        q(a)::PointMassFormConstraint(starting_point=(args...) -> [1.0])
    end
    init = @initialization begin
        q(s) = Dirichlet([2.0, 2.0])
        q(z) = Categorical([0.5, 0.5])
        q(b) = GammaShapeRate(2.0, 2.0)
    end
    args = (; model=compiled_gamma_mixture(), data=(y=[0.4, 0.7, 1.1, 1.8, 2.2, 2.8],),
        constraints, initialization=init, iterations=200, returnvars=KeepLast(), free_energy=true, session=nothing)
    baseline = infer(; args...)
    actual = infer(; args..., options=(runner=CompiledRunner(),))
    @test any(kernel -> kernel isa ReactiveMP.CompiledVariationalProductKernel,
        actual.model.metadata[:inference_runner].kernels)
    for name in (:a, :b, :z), statistic in (mean, var)
        @test statistic.(actual.posteriors[name]) ≈ statistic.(baseline.posteriors[name]) rtol=1e-4 atol=1e-6
    end
    @test mean(actual.posteriors[:s]) ≈ mean(baseline.posteriors[:s]) rtol=1e-4 atol=1e-6
    @test var(actual.posteriors[:s]) ≈ var(baseline.posteriors[:s]) rtol=1e-4 atol=1e-6
    @test actual.free_energy[end] ≈ baseline.free_energy[end] rtol=1e-4 atol=1e-6
end

@testitem "CompiledRunner: grouped Gaussian mixtures" begin
    @model function compiled_gmm(y)
        s ~ Beta(2.0, 2.0)
        m[1] ~ NormalMeanVariance(-2.0, 2.0)
        m[2] ~ NormalMeanVariance(2.0, 2.0)
        p[1] ~ GammaShapeRate(2.0, 2.0)
        p[2] ~ GammaShapeRate(2.0, 2.0)
        for i in eachindex(y)
            z[i] ~ Bernoulli(s)
            y[i] ~ NormalMixture(switch = z[i], m = m, p = p)
        end
    end
    init = @initialization begin
        q(s) = Beta(2.0, 2.0)
        q(m) = [NormalMeanVariance(-2.0, 1.0), NormalMeanVariance(2.0, 1.0)]
        q(p) = GammaShapeRate(2.0, 2.0)
    end
    y = [(-1)^i * 2.0 + 0.2sin(i) for i in 1:40]
    args = (; model = compiled_gmm(), data = (; y), initialization = init,
        constraints = MeanField(), iterations = 100, returnvars = KeepLast(),
        free_energy = true, session = nothing, disable_inference_error_hint = true)
    baseline = infer(; args...)
    actual = infer(; args..., options = (runner = CompiledRunner(),))
    @test mean(actual.posteriors[:s]) ≈ mean(baseline.posteriors[:s]) rtol=1e-4 atol=1e-6
    for name in (:m, :p, :z)
        @test mean.(actual.posteriors[name]) ≈ mean.(baseline.posteriors[name]) rtol=1e-4 atol=1e-6
        @test var.(actual.posteriors[name]) ≈ var.(baseline.posteriors[name]) rtol=1e-4 atol=1e-6
    end
    @test actual.free_energy[end] ≈ baseline.free_energy[end] rtol=1e-4 atol=1e-6
end

@testitem "CompiledRunner: Mixture log-scale annotations" begin
    @model function compiled_beta_mixture(y)
        selector ~ Bernoulli(0.7)
        in1 ~ Beta(4.0, 8.0)
        in2 ~ Beta(8.0, 4.0)
        θ ~ Mixture(switch = selector, inputs = [in1, in2])
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end
    args = (; model = compiled_beta_mixture(), data = (y = [1.0, 1.0, 0.0, 1.0],),
        annotations = LogScaleAnnotations(), iterations = 2, returnvars = KeepLast(),
        session = nothing, disable_inference_error_hint = true)
    baseline = infer(; args...)
    actual = infer(; args..., options = (runner = CompiledRunner(),))
    for name in keys(baseline.posteriors)
        @test mean(actual.posteriors[name]) ≈ mean(baseline.posteriors[name])
        @test var(actual.posteriors[name]) ≈ var(baseline.posteriors[name])
    end
end

@testitem "CompiledRunner: streaming autoupdates, bounded history, and restart" begin
    using Rocket
    @model function compiled_online(y, a, b)
        p ~ Beta(a, b)
        y ~ Bernoulli(p)
    end
    shapes(q) = (q.α, q.β)
    # Auto-update mappings receive Marginal wrappers, as in the reactive API.
    shapes(q::Marginal) = shapes(getdata(q))
    updates = @autoupdates begin
        a, b = shapes(q(p))
    end
    init = @initialization begin
        q(p) = Beta(2.0, 3.0)
    end
    args = (; model = compiled_online(), data = (y = [1.0, 0.0, 1.0, 1.0],),
        autoupdates = updates, initialization = init, iterations = 2, keephistory = 4,
        returnvars = [:p], historyvars = (p = KeepLast(),), free_energy = true,
        session = nothing)
    baseline = infer(; args...)
    engine = infer(; args..., options = (runner = CompiledRunner(),))
    @test engine isa RxInferenceEngine
    @test engine isa RxInfer.CompiledInferenceEngine
    @test engine.is_completed
    @test !engine.is_errored
    @test mean.(engine.history[:p]) ≈ mean.(baseline.history[:p])
    @test var.(engine.history[:p]) ≈ var.(baseline.history[:p])
    @test engine.free_energy_raw_history ≈ baseline.free_energy_raw_history

    source = Rocket.Subject(@NamedTuple{y::Float64})
    manual = infer(; model = compiled_online(), datastream = source, autoupdates = updates,
        initialization = init, iterations = 1, keephistory = 2, returnvars = [:p],
        options = (runner = CompiledRunner(),), autostart = false, uselock = true, session = nothing)
    actor = Rocket.keep(Any)
    subscription = Rocket.subscribe!(manual.posteriors[:p], actor)
    RxInfer.start(manual)
    Rocket.next!(source, (y = 1.0,))
    RxInfer.stop(manual)
    Rocket.next!(source, (y = 0.0,)) # stopped data must not update the program
    @test length(Rocket.getvalues(actor)) == 1
    RxInfer.start(manual)
    Rocket.next!(source, (y = 1.0,))
    Rocket.next!(source, (y = 0.0,))
    @test length(manual.history[:p]) == 2
    @test length(Rocket.getvalues(actor)) == 3
    @test mean(last(Rocket.getvalues(actor))) ≈ 4 / 8
    Rocket.complete!(source)
    @test manual.is_completed
    Rocket.unsubscribe!(subscription)
end

@testitem "CompiledRunner: message autoupdates, variable iterations, and failures" begin
    using Rocket
    @model function compiled_online_message(y, m, v)
        x ~ NormalMeanVariance(m, v)
        y ~ NormalMeanVariance(x, 1.0)
    end
    moments(message) = (mean(message), var(message))
    updates = @autoupdates begin
        m, v = moments(μ(x))
    end
    init = @initialization begin
        μ(x) = NormalMeanVariance(0.0, 1.0)
    end
    args = (; model=compiled_online_message(), data=(y=[0.2, 0.4, 0.6],),
        initialization=init, autoupdates=updates, iterations=2, keephistory=3,
        returnvars=[:x], historyvars=(x=KeepLast(),), session=nothing)
    reference = infer(; args...)
    actual = infer(; args..., options=(runner=CompiledRunner(),))
    @test mean.(actual.history[:x]) ≈ mean.(reference.history[:x])
    @test var.(actual.history[:x]) ≈ var.(reference.history[:x])

    source = Rocket.Subject(@NamedTuple{y::Float64})
    iterations = Ref(2)
    engine = infer(model=compiled_online_message(m=0.0, v=1.0), datastream=source,
        iterations=iterations, keephistory=2, historyvars=(x=KeepEach(),), returnvars=[:x],
        options=(runner=CompiledRunner(),), session=nothing)
    Rocket.next!(source, (y=0.5,))
    iterations[] = 3
    Rocket.next!(source, (y=0.7,))
    @test length.(engine.history[:x]) == [2, 3]
    Rocket.complete!(source)

    failing_source = Rocket.Subject(@NamedTuple{y::Float64})
    count = Ref(0)
    callbacks = (on_marginal_update=event -> begin
        count[] += 1
        count[] == 2 && error("injected posterior callback failure")
    end,)
    failed = infer(model=compiled_online_message(m=0.0, v=1.0), datastream=failing_source,
        iterations=1, keephistory=2, free_energy=true, returnvars=[:x], callbacks=callbacks,
        options=(runner=CompiledRunner(),), uselock=true, warn=false, session=nothing)
    Rocket.next!(failing_source, (y=0.5,))
    @test_throws ErrorException Rocket.next!(failing_source, (y=0.7,))
    @test failed.is_errored
    @test !failed.is_running
    program = failed.model.metadata[:inference_runner]
    @test program.failed
    sweeps = program.sweeps
    RxInfer.start(failed)
    Rocket.next!(failing_source, (y=1.0,))
    @test program.sweeps == sweeps
    @test length(failed.history[:x]) == 1
    @test length(failed.free_energy_raw_history) == 1

    args = (; model=compiled_online_message(m=0.0, v=1.0), data=(y=0.5,),
        iterations=3, returnvars=KeepLast(), options=(runner=CompiledRunner(),), session=nothing)
    failure = infer(; args..., callbacks=(after_iteration=event -> error("injected iteration failure"),),
        catch_exception=true, disable_inference_error_hint=true)
    @test RxInfer.iserror(failure)
    @test failure.model.metadata[:inference_runner].failed
    fresh = infer(; args...)
    @test RxInfer.issuccess(fresh)
    @test mean(fresh.posteriors[:x]) ≈ 0.25
end

@testitem "CompiledRunner: nonlinear Delta and observed arguments" begin
    delta_affine(x, a) = 2x + a
    delta_smooth(x, a) = x + a * sin(x)
    @model function compiled_delta(y, a, method, fn)
        x ~ NormalMeanVariance(0.0, 1.0)
        z ~ fn(x, a) where {meta = method}
        y ~ NormalMeanVariance(z, 0.5)
    end
    init = @initialization begin
        μ(x) = NormalMeanVariance(0.0, 1.0)
        μ(z) = NormalMeanVariance(0.0, 1.0)
    end
    for fn in (delta_affine, delta_smooth), method in (Linearization(), Unscented())
        args = (; model = compiled_delta(; fn, method), data = (y = 0.7, a = 0.2),
            initialization = init, iterations = 30, returnvars = KeepLast(),
            free_energy = true, session = nothing, disable_inference_error_hint = true)
        baseline = infer(; args...)
        actual = infer(; args..., options = (runner = CompiledRunner(),))
        for name in (:x, :z)
            @test mean(actual.posteriors[name]) ≈ mean(baseline.posteriors[name]) rtol=1e-4 atol=1e-6
            @test var(actual.posteriors[name]) ≈ var(baseline.posteriors[name]) rtol=1e-4 atol=1e-6
        end
        @test actual.free_energy[end] ≈ baseline.free_energy[end] rtol=1e-4 atol=1e-6
    end
end

@testitem "CompiledRunner: seeded nonlinear CVI compatibility" begin
    using Optimisers, Random, Statistics
    smooth_cvi(x) = x + 0.05sin(x)
    @model function compiled_cvi(y, method)
        x ~ NormalMeanVariance(0.0, 1.0)
        z ~ smooth_cvi(x) where {meta=method}
        y ~ NormalMeanVariance(z, 0.5)
    end
    init = @initialization begin
        μ(x) = NormalMeanVariance(0.0, 1.0)
        μ(z) = NormalMeanVariance(0.0, 1.0)
    end
    function run_cvi(seed, options)
        method = CVI(MersenneTwister(seed), 1000, 100, Optimisers.Descent(0.03))
        return infer(model=compiled_cvi(; method), data=(y=0.7,), initialization=init,
            iterations=20, free_energy=true, returnvars=KeepLast(), options=options, session=nothing)
    end
    # Different schedules consume random draws differently. Check the stochastic
    # approximation at a stated Monte Carlo tolerance, not bitwise trajectories.
    differences = Float64[]
    for seed in (11, 22, 33)
        reference = run_cvi(seed, (;))
        actual = run_cvi(seed, (runner=CompiledRunner(),))
        for name in (:x, :z)
            @test mean(actual.posteriors[name]) ≈ mean(reference.posteriors[name]) atol=0.08 rtol=0
            @test var(actual.posteriors[name]) ≈ var(reference.posteriors[name]) atol=0.03 rtol=0
        end
        push!(differences, mean(actual.posteriors[:x]) - mean(reference.posteriors[:x]))
        @test all(isfinite, actual.free_energy)
        @test !all(actual.model.metadata[:inference_runner].phase_parallel)
    end
    @test abs(mean(differences)) < 0.03
end

@testitem "CompiledRunner: missing data predictions and snapshots" begin
    @model function compiled_predict(y)
        x ~ NormalMeanVariance(0.0, 1.0)
        for i in eachindex(y)
            y[i] ~ NormalMeanVariance(x, 2.0)
        end
    end
    args = (; model = compiled_predict(), data = (y = [1.0, missing, -0.5],),
        iterations = 3, returnvars = (x = KeepEach(),), predictvars = (y = KeepEach(),),
        session = nothing, disable_inference_error_hint = true)
    baseline = infer(; args...)
    actual = infer(; args..., options = (runner = CompiledRunner(),))
    @test length(actual.posteriors[:x]) == 3
    @test length(actual.predictions[:y]) == 3
    @test mean.(actual.posteriors[:x]) ≈ mean.(baseline.posteriors[:x])
    @test mean.(actual.predictions[:y][end]) ≈ mean.(baseline.predictions[:y][end])
    @test var.(actual.predictions[:y][end]) ≈ var.(baseline.predictions[:y][end])
    shorthand = infer(model=compiled_predict(), data=(y=[1.0, missing, -0.5],),
        predictvars=KeepLast(), options=(runner=CompiledRunner(),), session=nothing)
    @test mean.(shorthand.predictions[:y]) ≈ mean.(baseline.predictions[:y][end])
end
