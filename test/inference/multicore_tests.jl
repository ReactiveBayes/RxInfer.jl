@testitem "MulticoreRunner: configuration, conjugacy, and deterministic waves" begin
    @test_throws ArgumentError MulticoreRunner(workers = 0)
    @test_throws ArgumentError MulticoreRunner(workers = Threads.nthreads() + 1)
    @test_throws ArgumentError MulticoreRunner(min_batch_size = 0)
    @test_throws ArgumentError MulticoreRunner(min_work_ns = -1)
    @test_throws ArgumentError convert(
        RxInfer.ReactiveMPInferenceOptions, (runner = :unknown,)
    )
    @test isnothing(
        RxInfer.getpostprocessor(
            convert(RxInfer.ReactiveMPInferenceOptions, (runner = nothing,))
        ),
    )
    @test ReactiveMP.multicore_readonly((symbol = :x, number = 1))
    @test !ReactiveMP.multicore_readonly((buffer = zeros(4),))
    @test !ReactiveMP.multicore_readonly((ref = Ref(1),))
    job = ReactiveMP.MulticoreMapJob{Nothing, typeof(identity), Int, Int}(
        nothing, identity, 7, false, nothing, nothing, false
    )
    @test job.result === nothing
    ReactiveMP.compute_multicore_job!(job)
    @test job.result === 7
    @test job.done
    mapping =
        (constraint, meta) -> ReactiveMP.MessageMapping(
            NormalMeanVariance,
            Val(:out),
            constraint,
            nothing,
            nothing,
            meta,
            nothing,
            nothing,
            nothing,
            nothing,
        )
    @test ReactiveMP.multicore_parallel_safe(
        mapping(Marginalisation(), nothing)
    )
    @test !ReactiveMP.multicore_parallel_safe(
        mapping(Marginalisation(), Ref(1))
    )
    @test !ReactiveMP.multicore_parallel_safe(mapping(Ref(1), nothing))
    @test_throws ArgumentError convert(
        RxInfer.ReactiveMPInferenceOptions,
        (runner = MulticoreRunner(), stream_postprocessors = nothing),
    )

    @model function multicore_normal(y)
        m ~ NormalMeanVariance(0.0, 1.0)
        p ~ GammaShapeRate(1.0, 1.0)
        for i in eachindex(y)
            y[i] ~ NormalMeanPrecision(m, p)
        end
    end
    constraints = @constraints begin
        q(m, p) = q(m)q(p)
    end
    initialization = @initialization begin
        q(m) = NormalMeanVariance(0.0, 1.0)
        q(p) = GammaShapeRate(1.0, 1.0)
    end
    args = (
        model = multicore_normal(),
        data = (y = sin.(1:100),),
        constraints = constraints,
        initialization = initialization,
        iterations = 30,
        free_energy = true,
        session = nothing,
        disable_inference_error_hint = true,
    )
    serial = infer(; args...)
    one = infer(;
        args...,
        options = (
            runner = MulticoreRunner(
                workers = 1, min_batch_size = 1, min_work_ns = 0
            ),
        ),
    )
    many = infer(;
        args...,
        options = (
            runner = MulticoreRunner(min_batch_size = 1, min_work_ns = 0),
        ),
    )
    @test RxInfer.issuccess(one) && RxInfer.issuccess(many)
    @test one.posteriors == many.posteriors
    @test one.free_energy == many.free_energy
    @test mean(serial.posteriors[:m][end]) ≈ mean(one.posteriors[:m][end])
    @test mean(serial.posteriors[:p][end]) ≈ mean(one.posteriors[:p][end])
    @test serial.free_energy[end] ≈ one.free_energy[end]
    state = many.model.metadata[:inference_runner]
    @test !state.active && !state.draining
    @test isempty(state.pending) && isempty(state.current)
    @test state.waves > 0
    if Threads.nthreads() > 1
        @test state.parallel_jobs > 0
    end
end

@testitem "MulticoreRunner: inference catches rule errors and configuration is reusable" begin
    @model function multicore_bad_probability(p)
        for i in eachindex(p)
            x[i] ~ Bernoulli(p[i])
            x[i] ~ Uninformative()
        end
    end
    config = MulticoreRunner(min_batch_size = 1)
    args = (
        model = multicore_bad_probability(),
        iterations = 1,
        options = (runner = config,),
        catch_exception = true,
        session = nothing,
        disable_inference_error_hint = true,
    )
    failed = infer(; args..., data = (p = fill(2.0, 64),))
    @test RxInfer.iserror(failed)
    state = failed.model.metadata[:inference_runner]
    @test !state.active && !state.draining
    @test isempty(state.pending) && isempty(state.current)
    recovered = infer(; args..., data = (p = fill(0.7, 64),))
    @test RxInfer.issuccess(recovered)
    @test recovered.model.metadata[:inference_runner] !== state
end

@testitem "MulticoreRunner: queued work, errors, and conservative fallback" begin
    using Base.Threads
    runner = ReactiveMP.instantiate_runner(
        MulticoreRunner(min_batch_size = 1, min_work_ns = 0)
    )
    source = Rocket.Subject(Int)
    outputs = Tuple{Int, Int}[]
    active = Atomic{Int}(0)
    # This bounded CPU workload also exercises overlapping worker execution.
    mapping = i -> begin
        atomic_add!(active, 1)
        acc = 0.0
        for k in 1:10000
            acc += sin(Float64(k + i))
        end
        atomic_sub!(active, 1)
        (i + (isfinite(acc) ? 0 : 1), threadid())
    end
    stream = ReactiveMP.runner_map(
        runner, Tuple{Int, Int}, source, mapping, true
    )
    subscription = subscribe!(stream, value -> push!(outputs, value))
    ReactiveMP.start_runner!(runner)
    for i in 1:100
        next!(source, i)
    end
    @test isempty(outputs)
    ReactiveMP.synchronize_runner!(runner)
    @test first.(outputs) == collect(1:100)
    @test active[] == 0
    if nthreads() > 1
        @test length(unique(last.(outputs))) > 1
    end
    ReactiveMP.stop_runner!(runner)
    unsubscribe!(subscription)

    source = Rocket.Subject(Int)
    ordered = Any[]
    stream = ReactiveMP.runner_map(runner, Int, source, identity, true)
    subscription = subscribe!(
        stream,
        lambda(
            Int;
            on_next = value -> push!(ordered, value),
            on_complete = () -> push!(ordered, :complete),
        ),
    )
    ReactiveMP.start_runner!(runner)
    foreach(i -> next!(source, i), 1:5)
    complete!(source)
    @test isempty(ordered)
    ReactiveMP.synchronize_runner!(runner)
    @test ordered == Any[1, 2, 3, 4, 5, :complete]
    ReactiveMP.stop_runner!(runner)
    unsubscribe!(subscription)

    source = Rocket.Subject(Int)
    outputs = Int[]
    stream = ReactiveMP.runner_map(
        runner,
        Int,
        source,
        i -> (i == 3 ? throw(ArgumentError("worker failure")) : i),
        true,
    )
    subscription = subscribe!(stream, value -> push!(outputs, value))
    ReactiveMP.start_runner!(runner)
    foreach(i -> next!(source, i), 1:10)
    @test_throws ArgumentError ReactiveMP.synchronize_runner!(runner)
    @test isempty(outputs)
    @test !runner.draining
    ReactiveMP.stop_runner!(runner)
    @test isempty(runner.pending) && isempty(runner.current)
    unsubscribe!(subscription)

    source = Rocket.Subject(Int)
    caller = threadid()
    stream = ReactiveMP.runner_map(
        runner,
        Int,
        source,
        i -> (threadid() == caller ? i : error("unsafe job moved to worker")),
        false,
    )
    outputs = Int[]
    subscription = subscribe!(stream, value -> push!(outputs, value))
    ReactiveMP.start_runner!(runner)
    foreach(i -> next!(source, i), 1:10)
    ReactiveMP.synchronize_runner!(runner)
    @test outputs == collect(1:10)
    ReactiveMP.stop_runner!(runner)
    unsubscribe!(subscription)
end

@testitem "MulticoreRunner: calibration executes each update exactly once" begin
    runner = ReactiveMP.instantiate_runner(
        MulticoreRunner(min_batch_size = 1, min_work_ns = typemax(Int))
    )
    source = Rocket.Subject(Int)
    calls = Threads.Atomic{Int}(0)
    outputs = Int[]
    mapping = i -> (Threads.atomic_add!(calls, 1); i * i)
    subscription = subscribe!(
        ReactiveMP.runner_map(runner, Int, source, mapping, true),
        value -> push!(outputs, value),
    )
    ReactiveMP.start_runner!(runner)
    for round in 1:2
        foreach(i -> next!(source, i), 1:100)
        ReactiveMP.synchronize_runner!(runner)
        @test calls[] == round * 100
    end
    @test outputs == repeat((1:100) .^ 2, 2)
    @test runner.parallel_jobs == 0
    @test runner.serial_jobs == 200
    ReactiveMP.stop_runner!(runner)
    unsubscribe!(subscription)
end

@testitem "MulticoreRunner: categorical hidden Markov model" begin
    @model function multicore_hmm(y)
        A ~ DirichletCollection([10.0 1.0; 1.0 10.0])
        B ~ DirichletCollection([10.0 1.0; 1.0 10.0])
        s0 ~ Categorical([0.5, 0.5])
        previous = s0
        for t in eachindex(y)
            s[t] ~ DiscreteTransition(previous, A)
            y[t] ~ DiscreteTransition(s[t], B)
            previous = s[t]
        end
    end
    constraints = @constraints begin
        q(s, s0, A, B) = q(s, s0)q(A)q(B)
    end
    initialization = @initialization begin
        q(A) = DirichletCollection([10.0 1.0; 1.0 10.0])
        q(B) = DirichletCollection([10.0 1.0; 1.0 10.0])
        q(s) = Categorical([0.5, 0.5])
    end
    data = (y = [i < 10 ? [1.0, 0.0] : [0.0, 1.0] for i in 1:20],)
    args = (
        model = multicore_hmm(),
        data = data,
        constraints = constraints,
        initialization = initialization,
        iterations = 60,
        returnvars = (s = KeepLast(), A = KeepLast(), B = KeepLast()),
        free_energy = true,
        session = nothing,
        disable_inference_error_hint = true,
    )
    reference = infer(; args...)
    one = infer(;
        args...,
        options = (runner = MulticoreRunner(workers = 1, min_batch_size = 1),),
    )
    many = infer(;
        args...,
        options = (
            runner = MulticoreRunner(min_batch_size = 1, min_work_ns = 0),
        ),
    )
    @test one.posteriors == many.posteriors
    @test one.free_energy == many.free_energy
    @test reference.free_energy[end] ≈ many.free_energy[end] rtol = 1e-6
    @test mean(reference.posteriors[:A]) ≈ mean(many.posteriors[:A]) rtol = 1e-5
end

@testitem "MulticoreRunner: loopy Gaussian grid and stack limiting" begin
    @model function multicore_grid(b, side)
        for i in eachindex(b)
            x[i] ~ NormalMeanPrecision(b[i] / 5, 5.0)
        end
        for row in 1:side, col in 1:side
            i = (row - 1) * side + col
            if row < side
                x[i + side] ~ GaussianCoupling(x[i], 1.0)
            end
            if col < side
                x[i + 1] ~ GaussianCoupling(x[i], 1.0)
            end
        end
    end
    initialization = @initialization begin
        μ(x) = NormalMeanVariance(0.0, 1e6)
    end
    side = 5
    b = sin.(1:(side ^ 2))
    args = (
        model = multicore_grid(side = side),
        data = (b = b,),
        initialization = initialization,
        iterations = 60,
        returnvars = (x = KeepLast(),),
        session = nothing,
        disable_inference_error_hint = true,
    )
    reference = infer(; args..., options = (limit_stack_depth = 50,))
    one = infer(;
        args...,
        options = (
            runner = MulticoreRunner(workers = 1), limit_stack_depth = 50
        ),
    )
    many = infer(;
        args...,
        options = (
            runner = MulticoreRunner(min_work_ns = 0), limit_stack_depth = 50
        ),
    )
    @test one.posteriors == many.posteriors
    @test mean.(reference.posteriors[:x]) ≈ mean.(many.posteriors[:x]) atol =
        1e-9
    @test var.(reference.posteriors[:x]) ≈ var.(many.posteriors[:x]) atol = 1e-9
    if Threads.nthreads() > 1
        @test many.model.metadata[:inference_runner].parallel_jobs > 0
    end
    for row in 1:side, col in 1:side
        i = (row - 1) * side + col
        x = mean.(many.posteriors[:x])
        residual = 5x[i] - b[i]
        row > 1 && (residual -= x[i - side])
        row < side && (residual -= x[i + side])
        col > 1 && (residual -= x[i - 1])
        col < side && (residual -= x[i + 1])
        @test abs(residual) < 1e-9
    end
end

@testitem "MulticoreRunner: structured chain, missing data, and nonlinear nodes" begin
    @model function multicore_chain(y)
        x[1] ~ NormalMeanVariance(0.0, 1.0)
        for i in eachindex(y)
            x[i + 1] ~ NormalMeanVariance(x[i], 0.2)
            y[i] ~ NormalMeanVariance(x[i + 1], 0.3)
        end
    end
    data = (y = Union{Missing, Float64}[0.1, 0.2, missing, 0.4, 0.3],)
    args = (
        model = multicore_chain(),
        data = data,
        iterations = 3,
        returnvars = (x = KeepLast(),),
        session = nothing,
        disable_inference_error_hint = true,
    )
    reference = infer(; args...)
    actual = infer(;
        args...,
        options = (
            runner = MulticoreRunner(min_batch_size = 1, min_work_ns = 0),
        ),
    )
    @test mean.(reference.posteriors[:x]) ≈ mean.(actual.posteriors[:x])
    @test var.(reference.posteriors[:x]) ≈ var.(actual.posteriors[:x])

    @model function multicore_nonlinear(y)
        for i in eachindex(y)
            x[i] ~ NormalMeanVariance(0.3, 1.0)
            z[i] := sin(x[i]) where {meta = DeltaMeta(method = Linearization())}
            y[i] ~ NormalMeanVariance(z[i], 0.2)
        end
    end
    args = (
        model = multicore_nonlinear(),
        data = (y = sin.(1:16),),
        iterations = 3,
        returnvars = (x = KeepLast(),),
        session = nothing,
        disable_inference_error_hint = true,
    )
    reference = infer(; args...)
    actual = infer(;
        args...,
        options = (
            runner = MulticoreRunner(min_batch_size = 1, min_work_ns = 0),
        ),
    )
    @test mean.(reference.posteriors[:x]) ≈ mean.(actual.posteriors[:x])
    @test var.(reference.posteriors[:x]) ≈ var.(actual.posteriors[:x])
end

@testitem "MulticoreRunner: streaming auto-updates" begin
    @model function multicore_stream(y, a, b)
        θ ~ Beta(a, b)
        y ~ Bernoulli(θ)
    end
    autoupdates = @autoupdates begin
        a, b = params(q(θ))
    end
    initialization = @initialization begin
        q(θ) = Beta(1.0, 1.0)
    end
    args = (
        model = multicore_stream(),
        data = (y = [true, false, true, true],),
        autoupdates = autoupdates,
        initialization = initialization,
        iterations = 1,
        keephistory = 4,
        historyvars = (θ = KeepLast(),),
        session = nothing,
    )
    reference = infer(; args...)
    actual = infer(;
        args...,
        options = (
            runner = MulticoreRunner(min_batch_size = 1, min_work_ns = 0),
        ),
    )
    @test reference.history == actual.history
    @test params(actual.history[:θ][end]) == (4.0, 2.0)
    @test !actual.model.metadata[:inference_runner].active
end
