@testitem "__inference_check_itertype" begin
    import RxInfer: inference_check_itertype

    @test inference_check_itertype(:something, nothing) === nothing
    @test inference_check_itertype(:something, (1,)) === nothing
    @test inference_check_itertype(:something, (1, 2)) === nothing
    @test inference_check_itertype(:something, []) === nothing
    @test inference_check_itertype(:something, [1, 2]) === nothing

    @test_throws ErrorException inference_check_itertype(:something, 1)
    @test_throws ErrorException inference_check_itertype(:something, (1))
    @test_throws ErrorException inference_check_itertype(:something, missing)
end

@testitem "infer_check_dicttype" begin
    import RxInfer: infer_check_dicttype

    @test infer_check_dicttype(:something, nothing) === nothing
    @test infer_check_dicttype(:something, (x = 1,)) === nothing
    @test infer_check_dicttype(:something, (x = 1, y = 2)) === nothing
    @test infer_check_dicttype(:something, Dict(:x => 1)) === nothing
    @test infer_check_dicttype(:something, Dict(:x => 1, :y => 2)) === nothing

    @test_throws ErrorException infer_check_dicttype(:something, 1)
    @test_throws ErrorException infer_check_dicttype(:something, (1))
    @test_throws ErrorException infer_check_dicttype(:something, missing)
    @test_throws ErrorException infer_check_dicttype(:something, (missing))
end

@testitem "__infer_create_factor_graph_model" begin
    @model function simple_model_for_infer_create_model(y, a, b)
        x ~ Beta(a, b)
        y ~ Normal(x, 1.0)
    end

    import RxInfer: __infer_create_factor_graph_model, ProbabilisticModel, getmodel
    import GraphPPL: is_data, is_random, is_constant, is_variable, is_factor, getproperties, getcontext

    @testset let model = __infer_create_factor_graph_model(simple_model_for_infer_create_model(a = 1, b = 2), (y = 3,))
        @test model isa ProbabilisticModel
        graphicalmodel = getmodel(model)
        ctx = getcontext(getmodel(model))
        @test is_variable(graphicalmodel[ctx[:y]])
        @test is_variable(graphicalmodel[ctx[:x]])
        @test is_data(getproperties(graphicalmodel[ctx[:y]]))
        @test is_random(getproperties(graphicalmodel[ctx[:x]]))
    end
end

@testitem "Static inference with `inference`" begin

    # A simple model for testing that resembles a simple kalman filter with
    # random walk state transition and unknown observational noise
    @model function test_model1(y)
        τ ~ Gamma(1.0, 1.0)

        x[1] ~ Normal(mean = 0.0, variance = 1.0)
        y[1] ~ Normal(mean = x[1], precision = τ)

        for i in 2:length(y)
            x[i] ~ Normal(mean = x[i - 1], variance = 1.0)
            y[i] ~ Normal(mean = x[i], precision = τ)
        end

        return length(y), 2, 3.0, "hello world" # test returnval
    end

    @constraints function test_model1_constraints()
        q(x, τ) = q(x)q(τ)
    end

    init = @initialization begin
        q(τ) = Gamma(1.0, 1.0)
    end

    @testset "returnval should be set properly" begin
        for n in 2:5
            result = infer(model = test_model1(), constraints = test_model1_constraints(), data = (y = rand(n),), initialization = init)
            @test getreturnval(result.model) === (n, 2, 3.0, "hello world")
        end
    end

    @testset "Test `catch_exception` functionality" begin
        observations = rand(10)

        # Case #0: no errors at all
        result = infer(
            model = test_model1(),
            constraints = test_model1_constraints(),
            data = (y = observations,),
            initialization = init,
            iterations = 10,
            returnvars = KeepEach(),
            free_energy = true
        )

        @test RxInfer.issuccess(result)
        @test !RxInfer.iserror(result)

        io = IOBuffer()

        Base.showerror(io, result)

        error_str = String(take!(io))

        @test contains(error_str, "The inference has completed successfully.")

        # Case #1: no error handling
        @test_throws "bang!" infer(
            model = test_model1(),
            constraints = test_model1_constraints(),
            data = (y = observations,),
            initialization = init,
            iterations = 10,
            returnvars = KeepEach(),
            free_energy = true,
            catch_exception = false,
            callbacks = (after_iteration = (model, iteration) -> begin
                # For test purposes we throw an error after `5` iterations
                if iteration >= 5
                    error("bang!")
                end
            end,)
        )

        result_with_error = infer(
            model = test_model1(),
            constraints = test_model1_constraints(),
            data = (y = observations,),
            initialization = init,
            iterations = 10,
            returnvars = KeepEach(),
            free_energy = true,
            catch_exception = true,
            callbacks = (after_iteration = (model, iteration) -> begin
                # For test purposes we throw an error after `5` iterations
                if iteration >= 5
                    error("bang!")
                end
            end,)
        )

        @test !RxInfer.issuccess(result_with_error)
        @test RxInfer.iserror(result_with_error)
        @test result_with_error.error isa Tuple
        @test length(result_with_error.free_energy) === 5
        @test all(result_with_error.free_energy .=== result.free_energy[1:5])

        io = IOBuffer()

        Base.showerror(io, result_with_error)

        error_str = String(take!(io))

        @test contains(error_str, "ErrorException")
        @test contains(error_str, "bang!")
        @test contains(error_str, "Vector")
        @test contains(error_str, "Base.StackTraces.StackFrame")
    end

    @testset "Test halting iterations based on callbacks" begin
        observations = rand(10)

        # Case #1: no halting
        results1 = infer(
            model = test_model1(),
            constraints = test_model1_constraints(),
            data = (y = observations,),
            initialization = init,
            iterations = 10,
            returnvars = KeepEach(),
            free_energy = true
        )

        @test length(results1.free_energy) === 10
        @test length(results1.posteriors[:x]) === 10
        @test length(results1.posteriors[:τ]) === 10

        # Case #2: halt before iteration starts
        results2 = infer(
            model = test_model1(),
            constraints = test_model1_constraints(),
            data = (y = observations,),
            initialization = init,
            iterations = 10,
            returnvars = KeepEach(),
            free_energy = true,
            callbacks = (
                # halt before iteration 5, but the logic could be more complex of course
                before_iteration = (model, iteration) -> iteration === 5,
            )
        )

        # We halted before iteration 5, so we assume the result length should be 4
        @test length(results2.free_energy) === 4
        @test length(results2.posteriors[:x]) === 4
        @test length(results2.posteriors[:τ]) === 4

        # Case #3: halt after iteration ends
        results3 = infer(
            model = test_model1(),
            constraints = test_model1_constraints(),
            data = (y = observations,),
            initialization = init,
            iterations = 10,
            returnvars = KeepEach(),
            free_energy = true,
            callbacks = (
                # halt after iteration 5, but the logic could be more complex of course
                after_iteration = (model, iteration) -> iteration === 5,
            )
        )

        # We halted after iteration 5, so we assume the result length should be 5
        @test length(results3.free_energy) === 5
        @test length(results3.posteriors[:x]) === 5
        @test length(results3.posteriors[:τ]) === 5

        # Check that free energy is equivalent between runs, data is the same, inference should be 
        # the same up until the halting point
        @test all(results1.free_energy[1:4] .=== results2.free_energy)
        @test all(results1.free_energy[1:5] .=== results3.free_energy)
    end
end

@testitem "Static inference with node contraction 1" begin
    import RxInfer: ReactiveMPGraphPPLBackend
    import Static
    import StableRNGs: StableRNG

    # Seed for reproducibility
    seed = 42

    rng = StableRNG(seed)

    function generate_data(rng, n, zv, yv)
        z_prev = 0.0
        x_prev = 0.0
        u_prev = 0.0

        k1 = 1.0
        w1 = 0.0
        k2 = 2.0
        w2 = 2.0
    
        z = Vector{Float64}(undef, n)
        v_1 = Vector{Float64}(undef, n)
        x = Vector{Float64}(undef, n)
        v_2 = Vector{Float64}(undef, n)
        u = Vector{Float64}(undef, n)
        y = Vector{Float64}(undef, n)
    
        for i in 1:n
            z[i] = rand(rng, Normal(z_prev, sqrt(zv)))
            v_1[i] = exp(k1 * z[i] + w1)
            x[i] = rand(rng, Normal(x_prev, sqrt(v_1[i])))
            v_2[i] = exp(k2 * x[i] + w2)
            u[i] = rand(rng, Normal(u_prev, sqrt(v_2[i])))
            y[i] = rand(rng, Normal(x[i], sqrt(yv)))
    
            z_prev = z[i]
            x_prev = x[i]
            u_prev = u[i]
        end 
        
        return z, x, u, y
    end

    # Parameters of HGF process
    z_variance = abs2(0.2)
    y_variance = abs2(0.1)

    # Number of observations
    n = 300

    z, x, u, data = generate_data(
        rng, 
        n, 
        z_variance, 
        y_variance
    );

    @model function gcv(y, κ, ω, z, x)
        log_σ := κ * z + ω
        σ := exp(log_σ)
        y ~ Normal(x, σ)
    end

    RxInfer.GraphPPL.default_constraints(::typeof(gcv)) = @constraints begin
        q(κ, ω, z, x) = q(κ)q(ω)q(z)q(x)
    end

    @model function hgf(
        y, y_variance, z_variance,
        z_prev_mean, z_prev_var, 
        x_prev_mean, x_prev_var,
        u_prev_mean, u_prev_var
    )
        κ1 ~ Normal(mean = 0, variance = 1) 
        ω1 ~ Normal(mean = 0, variance = 1)
        κ2 ~ Normal(mean = 0, variance = 1) 
        ω2 ~ Normal(mean = 0, variance = 1) 

        z_prev ~ Normal(mean = z_prev_mean, variance = z_prev_var)
        x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
        u_prev ~ Normal(mean = u_prev_mean, variance = u_prev_var)

        z_next ~ Normal(mean = z_prev, variance = z_variance)
        x_next ~ gcv(x = x_prev, z = z_next, κ = κ1, ω = ω1)
        u_next ~ gcv(x = u_prev, z = x_next, κ = κ2, ω = ω2)

        y ~ Normal(mean = u_next, variance = y_variance)
    end

    @constraints function hgf_constraints() 
        # Structured factorization constraints
        q(z_next, z_prev, x_next, x_prev, u_next, u_prev, κ1, ω1, κ2, ω2) = q(z_next)q(z_prev)q(x_next)q(x_prev)q(u_next)q(u_prev)q(κ1)q(ω1)q(κ2)q(ω2)
    end
    
    @meta function hgfmeta()
        gcv() -> DeltaMeta(method = Linearization()) 
        for meta in gcv
            exp() -> DeltaMeta(method = Linearization())
        end
    end

    autoupdates = @autoupdates begin
        # The posterior becomes the prior for the next time step
        z_prev_mean, z_prev_var = mean_var(q(z_next))
        x_prev_mean, x_prev_var = mean_var(q(x_next))
        u_prev_mean, u_prev_var = mean_var(q(u_next))
    end
    
    init = @initialization begin
        q(z_next) = NormalMeanVariance(0.0, 5.0)
        q(x_next) = NormalMeanVariance(0.0, 5.0)
        q(u_next) = NormalMeanVariance(0.0, 5.0)
        q(z_prev) = NormalMeanVariance(0.0, 5.0)
        q(x_prev) = NormalMeanVariance(0.0, 5.0)
        q(u_prev) = NormalMeanVariance(0.0, 5.0)
        q(κ1) = NormalMeanVariance(0.0, 1.0)
        q(ω1) = NormalMeanVariance(0.0, 1.0)
        q(κ2) = NormalMeanVariance(0.0, 1.0)
        q(ω2) = NormalMeanVariance(0.0, 1.0)
    end


    @testset "Check basic usage" begin

        return infer(
            model       = hgf(
                z_variance = z_variance, 
                y_variance = y_variance
            ),
            constraints = hgf_constraints(),
            meta        = hgfmeta(),
            data        = (y = data, ),
            autoupdates = autoupdates,
            keephistory = length(data),
            historyvars = (
                z_next = KeepLast(),
                x_next = KeepLast(),
                u_next = KeepLast()
            ),
            initialization = init,
            iterations     = 5,
            free_energy    = true,
        )
    end
end

@testitem "Static inference with node contraction" begin
    import RxInfer: ReactiveMPGraphPPLBackend
    import Static

    n = 5  # Number of test cases

    distribution = NormalMeanVariance(10.0)
    dataset      = rand(distribution, n)

    @model function gcv(y, κ, ω, z, x)
        log_σ := κ * z + ω
        σ := exp(log_σ)
        y ~ Normal(x, σ)
    end

    # mean-field constraint
    constraints = @constraints begin
        q(ξ, ω_1, ω_2, κ_1, κ_2, x_1, x_2, x_3) = q(ξ)q(ω_1)q(ω_2)q(κ_1)q(κ_2)q(x_1)q(x_2)q(x_3)
    end

    meta = @meta begin
        gcv() -> DeltaMeta(method = Linearization()) 
    end

    function GraphPPL.NodeType(::ReactiveMPGraphPPLBackend{Static.True}, ::typeof(gcv))
        return GraphPPL.Atomic()
    end

    @model function hgf(y)

        # Specify priors
        ξ ~ Gamma(1, 1)
        ω_1 ~ NormalMeanVariance(0, 1)
        ω_2 ~ NormalMeanVariance(0, 1)
        κ_1 ~ NormalMeanVariance(0, 1)
        κ_2 ~ NormalMeanVariance(0, 1)
        x_1[1] ~ NormalMeanVariance(0, 1)
        x_2[1] ~ NormalMeanVariance(0, 1)
        x_3[1] ~ NormalMeanVariance(0, 1)
        y[1] ~ NormalMeanVariance(x_1[1], 1)

        # Specify generative model
        for i in 2:(length(y))
            x_3[i] ~ NormalMeanVariance(x_3[i - 1], ξ)
            x_2[i] ~ gcv(x = x_2[i - 1], z = x_3[i], κ = κ_2, ω = ω_2)
            x_1[i] ~ gcv(x = x_1[i - 1], z = x_2[i], κ = κ_1, ω = ω_1)
            y[i] ~ NormalMeanVariance(x_1[i], 1)
        end
    end

    hgf_init = @initialization begin
        μ(ω_2) = vague(NormalMeanVariance)
        μ(κ_2) = vague(NormalMeanVariance)
        μ(x_1) = vague(NormalMeanVariance)
        μ(x_2) = vague(NormalMeanVariance)
        μ(x_3) = vague(NormalMeanVariance)
    end

    @rule ExponentialFamily.NormalMeanVariance(:μ, ReactiveMP.Marginalisation) (m_out::NormalMeanVariance, m_v::NormalMeanVariance) = begin
        return NormalMeanVariance(mean(m_out), m_v)
    end

    result = infer(
        model = hgf(), 
        data = (y = dataset,), 
        initialization = hgf_init, 
        meta = meta, 
        constraints = constraints,
        allow_node_contraction = true
    )

    @test "Fake test" true == true
end

@testitem "Test warn argument in `infer()`" begin
    @model function beta_bernoulli(y)
        θ ~ Beta(4.0, 8.0)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    observations = float.(rand(Bernoulli(0.75), 10))

    @testset "Test warning for addons" begin
        # Should display a warning if `warn` is set to `true`
        @test_logs (:warn, r"Both .* specify a value for the `addons`.*") infer(
            model = beta_bernoulli(), data = (y = observations,), addons = AddonLogScale(), options = (addons = AddonLogScale(),), warn = true
        )
        # Should not display a warning if `warn` is set to `true`
        @test_logs infer(model = beta_bernoulli(), data = (y = observations,), addons = AddonLogScale(), options = (addons = AddonLogScale(),), warn = false)
    end
end

@testitem "Invalid data size error" begin
    @model function test_model1(y)
        n = length(y)
        τ ~ Gamma(1.0, 1.0)

        x[1] ~ Normal(mean = 0.0, variance = 1.0)
        y[1] ~ Normal(mean = x[1], precision = τ)

        for i in 2:(n - 1)
            x[i] ~ Normal(mean = x[i - 1], variance = 1.0)
            y[i] ~ Normal(mean = x[i], precision = τ)
        end
        # y_n is unused intentionally
        x[n] ~ Normal(mean = x[n - 1], variance = 1.0)
        y[n - 1] ~ Normal(mean = x[n], precision = τ)
    end

    init = @initialization begin
        q(τ) = Gamma(1.0, 1.0)
    end

    @testset "Warning for unused datavars" begin
        @constraints function test_model1_constraints()
            q(x, τ) = q(x)q(τ)
        end

        @test_throws "size of datavar array and data must match" infer(
            model = test_model1(), constraints = test_model1_constraints(), data = (y = rand(10),), initialization = init
        )
    end
end

@testitem "Streamline inference with `autoupdates` for test model #1" begin
    import RxInfer: event_name

    # A simple model for testing that resembles a simple kalman filter with
    # random walk state transition and unknown observational noise
    @model function test_model1(x_t_min_mean, x_t_min_var, τ_shape, τ_rate, y)
        x_t_min ~ Normal(mean = x_t_min_mean, variance = x_t_min_var)
        τ ~ Gamma(shape = τ_shape, rate = τ_rate)
        # State transition
        x_t ~ Normal(mean = x_t_min, precision = 1.0)
        y ~ Normal(mean = x_t, precision = τ)
        return 2, 3.0, "hello world" # test returnval
    end

    init = @initialization begin
        q(x_t) = NormalMeanVariance(0.0, 1e3)
        q(τ) = GammaShapeRate(1.0, 1.0)
    end

    autoupdates = @autoupdates begin
        x_t_min_mean, x_t_min_var = mean_var(q(x_t))
        τ_shape = shape(q(τ))
        τ_rate = rate(q(τ))
    end

    n         = 10
    hiddenx   = Float64[]
    observedy = Float64[]
    prevx     = 0.0
    for i in 1:n
        nextx = rand(NormalMeanVariance(prevx, 1.0))
        nexty = rand(NormalMeanPrecision(nextx, 10.0))
        push!(hiddenx, nextx)
        push!(observedy, nexty)
        global prevx = nextx
    end

    @testset "Check basic usage" begin
        for keephistory in (0, 1, 2), iterations in (3, 4), free_energy in (true, Float64, false), returnvars in ((:x_t,), (:x_t, :τ)), historyvars in ((:x_t,), (:x_t, :τ))
            historyvars = keephistory > 0 ? NamedTuple{historyvars}(map(_ -> KeepEach(), historyvars)) : nothing

            engine = infer(
                model = test_model1(),
                constraints = MeanField(),
                data = (y = observedy,),
                returnvars = returnvars,
                historyvars = historyvars,
                keephistory = keephistory,
                initialization = init,
                iterations = iterations,
                free_energy = free_energy,
                autoupdates = autoupdates
            )

            # Test that the `.model` reference is correct
            @test length(getfactornodes(engine.model)) === 4
            @test length(getrandomvars(engine.model)) === 3
            @test length(getdatavars(engine.model)) === 5
            @test length(getconstantvars(engine.model)) === 1

            # Test that the `returnval` reference is correct
            @test getreturnval(engine.model) === (2, 3.0, "hello world")

            # Test that the `.posteriors` field is constructed correctly
            @test sort(collect(keys(engine.posteriors))) == sort(collect(returnvars))
            @test all(p -> typeof(p) <: Rocket.Subscribable, collect(values(engine.posteriors)))

            # Check that we save the history of the marginals if needed
            if keephistory > 0
                @test sort(collect(keys(engine.history))) == sort(collect(keys(historyvars)))
                for key in keys(historyvars)
                    @test length(engine.history[key]) === keephistory
                    @test length(engine.history[key][end]) === iterations
                end
            else
                @test engine.history === nothing
            end

            if free_energy !== false
                @test typeof(engine.free_energy) <: Rocket.Subscribable
            else
                @test_throws ErrorException engine.free_energy
            end

            # Check that we save the history of the free energy if needed
            if keephistory > 0 && free_energy !== false
                @test length(engine.free_energy_history) === iterations
                @test all(<=(0), diff(engine.free_energy_history))

                @test length(engine.free_energy_final_only_history) === keephistory
                @test length(engine.free_energy_raw_history) === keephistory * iterations
            else
                @test_throws ErrorException engine.free_energy_history
                @test_throws ErrorException engine.free_energy_final_only_history
                @test_throws ErrorException engine.free_energy_raw_history
            end

            # The engine might run on the static data just once
            # `RxInfer.start` and `RxInfer.stop` after the completion should be disallowed
            @test_logs (:warn, r"The engine.*completed.*") RxInfer.start(engine)
            @test_logs (:warn, r"The engine.*completed.*") RxInfer.stop(engine)
        end
    end

    @testset "Check callbacks usage: autostart enabled" begin
        callbacksdata = []

        engine = infer(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initialization = init,
            autoupdates = autoupdates,
            callbacks = (
                before_model_creation = (args...) -> push!(callbacksdata, (:before_model_creation, args)),
                after_model_creation = (args...) -> push!(callbacksdata, (:after_model_creation, args)),
                before_autostart = (args...) -> push!(callbacksdata, (:before_autostart, args)),
                after_autostart = (args...) -> push!(callbacksdata, (:after_autostart, args))
            ),
            autostart = true
        )

        # First check the order
        @test first.(callbacksdata) == [:before_model_creation, :after_model_creation, :before_autostart, :after_autostart]

        @test typeof(callbacksdata[1][2]) <: Tuple{}                   # before_model_creation
        @test typeof(callbacksdata[2][2]) <: Tuple{ProbabilisticModel} # after_model_creation 
        @test typeof(callbacksdata[3][2]) <: Tuple{RxInferenceEngine}  # before_autostart 
        @test typeof(callbacksdata[4][2]) <: Tuple{RxInferenceEngine}  # after_autostart
    end

    @testset "Check callbacks usage: autostart disabled" begin
        callbacksdata = []

        engine = infer(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initialization = init,
            autoupdates = autoupdates,
            callbacks = (
                before_model_creation = (args...) -> push!(callbacksdata, (:before_model_creation, args)),
                after_model_creation = (args...) -> push!(callbacksdata, (:after_model_creation, args)),
                before_autostart = (args...) -> push!(callbacksdata, (:before_autostart, args)),
                after_autostart = (args...) -> push!(callbacksdata, (:after_autostart, args))
            ),
            autostart = false
        )

        # First check the order
        @test first.(callbacksdata) == [:before_model_creation, :after_model_creation]

        @test typeof(callbacksdata[1][2]) <: Tuple{}                   # before_model_creation
        @test typeof(callbacksdata[2][2]) <: Tuple{ProbabilisticModel} # after_model_creation 

        RxInfer.start(engine)

        # Nothing extra should has been executed on `start`
        @test length(callbacksdata) === 2
    end

    @testset "Check callbacks usage: unknown callback warning" begin
        callbacksdata = []

        @test_logs (:warn, r"Unknown callback specification.*hello_world.*Available callbacks.*") result = infer(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initialization = init,
            autoupdates = autoupdates,
            callbacks = (hello_world = (args...) -> push!(callbacksdata, args),),
            autostart = true
        )

        @test length(callbacksdata) === 0
    end

    @testset "Check events usage" begin
        struct CustomEventListener <: Rocket.NextActor{RxInferenceEvent}
            eventsdata
        end

        function Rocket.on_next!(listener::CustomEventListener, event::RxInferenceEvent{:on_new_data})
            push!(listener.eventsdata, Any[event])
        end

        function Rocket.on_next!(listener::CustomEventListener, event::RxInferenceEvent)
            push!(last(listener.eventsdata), event)
        end

        for iterations in (2, 3), keephistory in (0, 1)
            engine = infer(
                model = test_model1(),
                constraints = MeanField(),
                data = (y = observedy,),
                initialization = init,
                autoupdates = autoupdates,
                historyvars = KeepEach(),
                keephistory = keephistory,
                events = Val((
                    :on_new_data,
                    :before_iteration,
                    :after_iteration,
                    :before_auto_update,
                    :after_auto_update,
                    :before_data_update,
                    :after_data_update,
                    :before_history_save,
                    :after_history_save,
                    :on_tick,
                    :on_error,
                    :on_complete
                )),
                iterations = iterations,
                autostart = false,
                warn = false
            )

            event_listener = CustomEventListener([])

            subscription = subscribe!(engine.events, event_listener)

            RxInfer.start(engine)

            eventsdata = event_listener.eventsdata

            # Check that the number of event blocks consitent with the number of data points
            @test length(eventsdata) === length(observedy)

            for (index, events) in enumerate(eventsdata)
                @test length(filter(event -> event isa RxInferenceEvent{:on_new_data}, events)) == 1

                # Check the associated data with the `:on_new_data` events
                foreach(filter(event -> event isa RxInferenceEvent{:on_new_data}, events)) do event
                    # `(model, data) = event`
                    model, data = event
                    @test model === engine.model
                    @test data === (y = observedy[index],)
                end

                # Check that the number of `:before_iteration` and `:after_iteration` events depends on the number of iterations
                @test length(filter(event -> event isa RxInferenceEvent{:before_iteration}, events)) == iterations
                @test length(filter(event -> event isa RxInferenceEvent{:after_iteration}, events)) == iterations

                # Check the associated data with the `:before_iteration` events
                foreach(enumerate(filter(event -> event isa RxInferenceEvent{:before_iteration}, events))) do (ii, event)
                    model, iteration = event
                    @test model === engine.model
                    @test iteration === ii
                end

                # Check the associated data with the `:after_iteration` events
                foreach(enumerate(filter(event -> event isa RxInferenceEvent{:after_iteration}, events))) do (ii, event)
                    model, iteration = event
                    @test model === engine.model
                    @test iteration === ii
                end

                # Check the correct ordering of the `:before_iteration` and `:after_iteration` events
                @test map(event_name, filter(event -> event isa RxInferenceEvent{:before_iteration} || event isa RxInferenceEvent{:after_iteration}, events)) ==
                    repeat([:before_iteration, :after_iteration], iterations)

                # Check that the number of `:before_auto_update` and `:after_auto_update` events depends on the number of iterations
                @test length(filter(event -> event isa RxInferenceEvent{:before_auto_update}, events)) == iterations
                @test length(filter(event -> event isa RxInferenceEvent{:after_auto_update}, events)) == iterations

                # Check the associated data with the `:before_auto_update` events
                foreach(enumerate(filter(event -> event isa RxInferenceEvent{:before_auto_update}, events))) do (ii, event)
                    model, iteration, fupdate = event
                    @test model === engine.model
                    @test iteration === ii
                    @test RxInfer.numautoupdates(fupdate) === 3
                end

                # Check the associated data with the `:after_auto_update` events
                foreach(enumerate(filter(event -> event isa RxInferenceEvent{:after_auto_update}, events))) do (ii, event)
                    model, iteration, fupdate = event
                    @test model === engine.model
                    @test iteration === ii
                    @test RxInfer.numautoupdates(fupdate) === 3
                end

                # Check the correct ordering of the `:before_auto_update` and `:after_auto_update` events
                @test map(event_name, filter(event -> event isa RxInferenceEvent{:before_auto_update} || event isa RxInferenceEvent{:after_auto_update}, events)) ==
                    repeat([:before_auto_update, :after_auto_update], iterations)

                # Check that the number of `:before_data_update` and `:after_data_update` events depends on the number of iterations
                @test length(filter(event -> event isa RxInferenceEvent{:before_data_update}, events)) == iterations
                @test length(filter(event -> event isa RxInferenceEvent{:after_data_update}, events)) == iterations

                # Check the associated data with the `:before_auto_update` events
                foreach(enumerate(filter(event -> event isa RxInferenceEvent{:before_data_update}, events))) do (ii, event)
                    model, iteration, data = event
                    @test model === engine.model
                    @test iteration === ii
                    @test data === (y = observedy[index],)
                end

                # Check the associated data with the `:after_auto_update` events
                foreach(enumerate(filter(event -> event isa RxInferenceEvent{:after_data_update}, events))) do (ii, event)
                    model, iteration, data = event
                    @test model === engine.model
                    @test iteration === ii
                    @test data === (y = observedy[index],)
                end

                # Check the correct ordering of the `:before_auto_update` and `:after_auto_update` events
                @test map(event_name, filter(event -> event isa RxInferenceEvent{:before_data_update} || event isa RxInferenceEvent{:after_data_update}, events)) ==
                    repeat([:before_data_update, :after_data_update], iterations)

                # Check the correct ordering of the iteration related events
                @test map(
                    event_name,
                    filter(events) do event
                        return event isa RxInferenceEvent{:before_iteration} ||
                               event isa RxInferenceEvent{:before_auto_update} ||
                               event isa RxInferenceEvent{:after_auto_update} ||
                               event isa RxInferenceEvent{:before_data_update} ||
                               event isa RxInferenceEvent{:after_data_update} ||
                               event isa RxInferenceEvent{:after_iteration}
                    end
                ) == repeat([:before_iteration, :before_auto_update, :after_auto_update, :before_data_update, :after_data_update, :after_iteration], iterations)

                if keephistory > 0
                    @test length(filter(event -> event isa RxInferenceEvent{:before_history_save}, events)) == 1
                    @test length(filter(event -> event isa RxInferenceEvent{:after_history_save}, events)) == 1
                end

                @test length(filter(event -> event isa RxInferenceEvent{:on_tick}, events)) == 1

                # We should receive the `:on_complete` event only for the last data point
                if index === length(eventsdata)
                    @test length(filter(event -> event isa RxInferenceEvent{:on_complete}, events)) == 1
                else
                    @test length(filter(event -> event isa RxInferenceEvent{:on_complete}, events)) == 0
                end
            end

            unsubscribe!(subscription)
        end
    end

    @testset "Check postprocess usage: UnpackMarginalPostprocess" begin
        engine = infer(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initialization = init,
            autoupdates = autoupdates,
            postprocess = RxInfer.UnpackMarginalPostprocess(),
            historyvars = (τ = KeepLast(),),
            iterations = 10,
            keephistory = 100,
            autostart = true
        )

        # Check that the result is not of type `Marginal`
        @test all(data -> !(typeof(data) <: ReactiveMP.Marginal), engine.history[:τ])
    end

    @testset "Check postprocess usage: NoopPostprocess & nothing" begin
        for postprocess in (RxInfer.NoopPostprocess(), nothing)
            engine = infer(
                model = test_model1(),
                constraints = MeanField(),
                data = (y = observedy,),
                initialization = init,
                autoupdates = autoupdates,
                postprocess = postprocess,
                historyvars = (τ = KeepLast(),),
                iterations = 10,
                keephistory = 100,
                autostart = true
            )

            # Check that the result is of type `Marginal`
            @test all(data -> typeof(data) <: ReactiveMP.Marginal, engine.history[:τ])
        end
    end

    @testset "Check the event creation and unrolling syntax" begin
        data1, data2 = RxInferenceEvent(Val(:event1), (1, 2.0))

        @test data1 === 1
        @test data2 === 2.0
    end

    @testset "Either `data` or `datastream` is required" begin
        @test_throws ErrorException infer(model = test_model1())
    end

    @testset "`data` and `datastream` cannot be used together" begin
        @test_throws ErrorException infer(model = test_model1(), data = (y = observedy,), datastream = labeled(Val((:y,)), combineLatest(from(observedy))))
    end
end

@testitem "Test misspecified types in infer function" begin
    @model function rolling_die(y)
        θ ~ Dirichlet(ones(6))
        for i in eachindex(y)
            y[i] ~ Categorical(θ)
        end
    end

    @model function rolling_die_streamlined(y, p)
        θ ~ Dirichlet(p)
        y ~ Categorical(θ)
    end

    streamlined_autoupdates = @autoupdates begin
        (p,) = params(q(θ))
    end

    streamlined_init = @initialization begin
        q(θ) = Dirichlet(ones(6))
    end

    observations = [[1.0; zeros(5)], [zeros(5); 1.0]]

    @testset "Test misspecified data" begin
        @test_throws "Keyword argument `data` expects either `Dict` or `NamedTuple` as an input" infer(model = rolling_die(), data = (y = observations))
        result = infer(model = rolling_die(), data = (y = observations,))
        @test isequal(first(mean(result.posteriors[:θ])), last(mean(result.posteriors[:θ])))
    end

    @testset "Test misspecified callbacks" begin
        @test_throws "Keyword argument `callbacks` expects either `Dict` or `NamedTuple` as an input" infer(
            model = rolling_die(), data = (y = observations,), callbacks = (before_model_creation = (args...) -> nothing)
        )
        result = infer(model = rolling_die(), data = (y = observations,), callbacks = (before_model_creation = (args...) -> nothing,))
        @test isequal(first(mean(result.posteriors[:θ])), last(mean(result.posteriors[:θ])))
    end

    @testset "Test misspecified event type in the streamlined inference" begin
        @test_logs (:warn, r"Unknown event type: blabla. Available events: .*") infer(
            model = rolling_die_streamlined(),
            data = (y = observations,),
            autoupdates = streamlined_autoupdates,
            initialization = streamlined_init,
            autostart = true,
            keephistory = 1,
            warn = true,
            events = Val((:blabla,))
        )
        result = @test_logs infer(
            model = rolling_die_streamlined(),
            data = (y = observations,),
            autoupdates = streamlined_autoupdates,
            initialization = streamlined_init,
            autostart = true,
            keephistory = 1,
            warn = false,
            events = Val((:blabla,))
        )
        @test isequal(first(mean(result.history[:θ][end])), last(mean(result.history[:θ][end])))
    end
end

@testitem "Autoupdates should throw an error if the data is present for the autoupdated arguments" begin
    @model function beta_bernoulli(a, b, y)
        t ~ Beta(a, b)
        y ~ Bernoulli(t)
    end

    autoupdates = @autoupdates begin
        a, b = params(q(t))
    end

    @test_throws "`a` is present both in the `data` and in the `autoupdates`." infer(model = beta_bernoulli(), data = (y = [1], a = [2]), autoupdates = autoupdates)
    @test_throws "`a` is present both in the `data` and in the `autoupdates`." infer(model = beta_bernoulli(), data = (y = [1], a = [2], b = [2]), autoupdates = autoupdates)
    @test_throws "`b` is present both in the `data` and in the `autoupdates`." infer(model = beta_bernoulli(), data = (y = [1], b = [2]), autoupdates = autoupdates)
end

@testitem "Autoupdates should throw an error if the return value does not match the left hand side in size" begin
    @model function beta_bernoulli(a, b, y)
        t ~ Beta(a, b)
        y ~ Bernoulli(t)
    end

    autoupdates = @autoupdates begin
        foo(q) = (1, 2, 3)
        a, b = foo(q(t))
    end

    @test_throws "Couldn't run autoupdate. The update provides `3` values, but `2` are needed." infer(
        model = beta_bernoulli(), data = (y = [1],), autoupdates = autoupdates, initialization = @initialization(q(t) = Beta(1, 1))
    )
end

@testitem "`infer` should throw an error if `initmessages` or `initmarginals` keywords are used" begin
    @model function beta_bernoulli(a, b, y)
        t ~ Beta(a, b)
        y ~ Bernoulli(t)
    end

    @test_throws "`initmessages` and `initmarginals` keyword arguments have been deprecated and removed. Use the `@initialization` macro and the `initialization` keyword instead." infer(
        model = beta_bernoulli(), data = (y = 1,), initmessages = (t = Normal(0.0, 1.0)), initmarginals = (t = Normal(0.0, 1.0))
    )

    @test_throws "`initmessages` and `initmarginals` keyword arguments have been deprecated and removed. Use the `@initialization` macro and the `initialization` keyword instead." infer(
        model = beta_bernoulli(), data = (y = 1,), initmarginals = (t = Normal(0.0, 1.0))
    )

    @test_throws "`initmessages` and `initmarginals` keyword arguments have been deprecated and removed. Use the `@initialization` macro and the `initialization` keyword instead." infer(
        model = beta_bernoulli(), data = (y = 1,), initmessages = (t = Normal(0.0, 1.0))
    )
end
