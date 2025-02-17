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
        y ~ Normal(mean = x, var = 1.0)
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
        τ ~ Gamma(shape = 1.0, rate = 1.0)

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

@testitem "Static inference with node contraction" begin
    import RxInfer: ReactiveMPGraphPPLBackend
    import Static

    n = 6  # Number of test cases

    distribution = NormalMeanVariance(0.0, 1.0)
    dataset      = rand(distribution, n)

    @model function gcv(y, x, z, κ, ω)
        log_σ := κ * z + ω
        σ := exp(log_σ)
        y ~ Normal(mean = x, precision = σ)
    end

    @node typeof(gcv) Stochastic [y, x, z, κ, ω]

    RxInfer.ReactiveMP.default_meta(::typeof(gcv)) = RxInfer.ReactiveMP.default_meta(GCV)

    @rule typeof(gcv)(:y, Marginalisation) (q_x::Any, q_z::Any, q_κ::Any, q_ω::Any, meta::Any) = begin
        return @call_rule GCV(:y, Marginalisation) (q_x = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta)
    end

    @rule typeof(gcv)(:x, Marginalisation) (q_y::Any, q_z::Any, q_κ::Any, q_ω::Any, meta::Any) = begin
        return @call_rule GCV(:x, Marginalisation) (q_y = q_y, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta)
    end

    @rule typeof(gcv)(:ω, Marginalisation) (q_y::Any, q_x::Any, q_z::Any, q_κ::Any, meta::Any) = begin
        return @call_rule GCV(:ω, Marginalisation) (q_y = q_y, q_x = q_x, q_z = q_z, q_κ = q_κ, meta = meta)
    end

    @rule typeof(gcv)(:z, Marginalisation) (q_y::Any, q_x::Any, q_κ::Any, q_ω::Any, meta::Any) = begin
        return @call_rule GCV(:z, Marginalisation) (q_y = q_y, q_x = q_x, q_κ = q_κ, q_ω = q_ω, meta = meta)
    end

    @rule typeof(gcv)(:κ, Marginalisation) (q_y::Any, q_x::Any, q_z::Any, q_ω::Any, meta::Any) = begin
        return @call_rule GCV(:κ, Marginalisation) (q_y = q_y, q_x = q_x, q_z = q_z, q_ω = q_ω, meta = meta)
    end

    @average_energy typeof(gcv) (q_y::Any, q_x::Any, q_z::Any, q_κ::Any, q_ω::Any, meta::Union{<:GCVMetadata, Nothing}) = begin
        y_mean, y_var = mean_var(q_y)
        x_mean, x_var = mean_var(q_x)
        z_mean, z_var = mean_var(q_z)
        κ_mean, κ_var = mean_var(q_κ)
        ω_mean, ω_var = mean_var(q_ω)

        ksi = (κ_mean^2) * z_var + (z_mean^2) * κ_var + κ_var * z_var
        psi = (y_mean - x_mean)^2 + y_var + x_var
        A = exp(-ω_mean + ω_var / 2)
        B = exp(-κ_mean * z_mean + ksi / 2)

        (log(2π) + (z_mean * κ_mean + ω_mean) + (psi * A * B)) / 2
    end

    @model function hgf_1(y)
        ω ~ NormalMeanVariance(0, 1)
        κ ~ NormalMeanVariance(1, 1)
        x_0 ~ NormalMeanVariance(0, 1)
        z[1] ~ NormalMeanVariance(0, 1)
        x[1] ~ gcv(x = x_0, z = z[1], κ = κ, ω = ω)
        y[1] ~ NormalMeanVariance(x[1], 1)

        for i in 2:length(y)
            z[i] ~ NormalMeanPrecision(z[i - 1], 1)
            x[i] ~ gcv(x = x[i - 1], z = z[i], κ = κ, ω = ω)
            y[i] ~ NormalMeanVariance(x[i], 1)
        end
    end

    @initialization function hgf_1_initialization()
        q(ω) = NormalMeanVariance(0, 1)
        q(κ) = NormalMeanVariance(1, 1)
        q(z) = NormalMeanVariance(0, 1)
        q(x) = NormalMeanVariance(0, 1)
    end

    result_1 = infer(model = hgf_1(), data = (y = dataset,), initialization = hgf_1_initialization(), constraints = MeanField(), allow_node_contraction = true, free_energy = true)

    @test all(!isnan, mean.(result_1.posteriors[:x]))
    @test all(!isnan, var.(result_1.posteriors[:x]))
    @test all(<=(0), diff(result_1.free_energy))

    @model function hgf_2(y)

        # Specify priors
        ω_1 ~ NormalMeanVariance(0, 1)
        ω_2 ~ NormalMeanVariance(0, 1)
        κ_1 ~ NormalMeanVariance(1, 1)
        κ_2 ~ NormalMeanVariance(1, 1)
        x_1[1] ~ NormalMeanVariance(0, 1)
        x_2[1] ~ NormalMeanVariance(0, 1)
        x_3[1] ~ NormalMeanVariance(0, 1)
        y[1] ~ NormalMeanVariance(x_1[1], 1)

        # Specify generative model
        for i in 2:(length(y))
            x_3[i] ~ NormalMeanPrecision(x_3[i - 1], 1)
            x_2[i] ~ gcv(x = x_2[i - 1], z = x_3[i], κ = κ_2, ω = ω_2)
            x_1[i] ~ gcv(x = x_1[i - 1], z = x_2[i], κ = κ_1, ω = ω_1)
            y[i] ~ NormalMeanVariance(x_1[i], 1)
        end
    end

    @initialization function hgf_2_initialization()
        q(ω_1) = vague(NormalMeanVariance)
        q(κ_1) = vague(NormalMeanVariance)
        q(ω_2) = vague(NormalMeanVariance)
        q(κ_2) = vague(NormalMeanVariance)
        q(x_1) = vague(NormalMeanVariance)
        q(x_2[1:2:n]) = vague(NormalMeanVariance)
        q(x_3) = vague(NormalMeanVariance)
    end

    result_2 = infer(model = hgf_2(), data = (y = dataset,), initialization = hgf_2_initialization(), constraints = MeanField(), allow_node_contraction = true)

    @test result_2.posteriors[:x_1] isa Vector{<:NormalDistributionsFamily}
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
        τ ~ Gamma(shape = 1.0, rate = 1.0)

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

@testitem "Unsupported functional forms (e.g. `ProductOf`) should display the name of the variable and suggestions" begin
    struct DistributionA
        a
    end
    struct DistributionB
        b
    end
    struct LikelihoodDistribution
        input
    end

    @node DistributionA Stochastic [out, a]
    @node DistributionB Stochastic [out, b]
    @node LikelihoodDistribution Stochastic [out, input]

    @rule DistributionA(:out, Marginalisation) (q_a::Any,) = DistributionA(mean(q_a))
    @rule DistributionB(:out, Marginalisation) (q_b::Any,) = DistributionB(mean(q_b))
    @rule LikelihoodDistribution(:input, Marginalisation) (q_out::Any,) = LikelihoodDistribution(mean(q_out))

    @model function invalid_product_posterior(out)
        θ ~ DistributionA(1.0)
        out ~ LikelihoodDistribution(θ)
    end

    # Product of `DistributionA` & `LikelihoodDistribution` in the posterior
    P = typeof(prod(GenericProd(), DistributionA(1.0), LikelihoodDistribution(1.0))) # the actual order may change though
    @test_throws """
    The expression `q(θ)` has an undefined functional form of type `$(P)`. 
    This is likely because the inference backend does not support the product of these distributions. 
    As a result, `RxInfer` cannot compute key quantities such as the `mean` or `var` of `q(θ)`.

    Possible solutions:
    - Alter model specification to ensure the prior is conjugate (see https://en.wikipedia.org/wiki/Conjugate_prior).
    - Implement the `BayesBase.prod` method (refer to the `BayesBase` documentation for guidance).
    - Use a functional form constraint to specify the posterior form with the `@constraints` macro. For example:
    ```julia
    using ExponentialFamilyProjection

    @constraints begin
        q(θ) :: ProjectedTo(NormalMeanVariance)
    end
    ```
    Refer to the documentation for more details on functional form constraints.
    """ result = infer(model = invalid_product_posterior(), data = (out = 1.0,))

    # Product of `DistributionA` & `DistributionB` in the message
    @model function invalid_product_message(out)
        input[1] ~ DistributionA(1.0)
        input[1] ~ DistributionB(1.0)
        θ ~ DistributionA(input[1])
        out ~ LikelihoodDistribution(θ)
    end

    T = typeof(prod(GenericProd(), DistributionB(1.0), DistributionA(1.0))) # the actual order may change though
    @test_throws """
    The expression `μ(input[1])` has an undefined functional form of type `$(T)`. 
    This is likely because the inference backend does not support the product of these distributions. 
    As a result, `RxInfer` cannot compute key quantities such as the `mean` or `var` of `μ(input[1])`.

    Possible solutions:
    - Alter model specification to ensure the prior is conjugate (see https://en.wikipedia.org/wiki/Conjugate_prior).
    - Implement the `BayesBase.prod` method (refer to the `BayesBase` documentation for guidance).
    - Use a functional form constraint to specify the posterior form with the `@constraints` macro. For example:
    ```julia
    using ExponentialFamilyProjection

    @constraints begin
        μ(input) :: ProjectedTo(NormalMeanVariance)
    end
    ```
    Refer to the documentation for more details on functional form constraints.
    """ result = infer(model = invalid_product_message(), data = (out = 1.0,), returnvars = (θ = KeepEach(),))
end

@testitem "`infer` with UnfactorizedData" begin
    using RxInfer

    @model function pred_model(p_s_t, y, goal, p_B, A)
        s[1] ~ p_s_t
        B ~ p_B
        y[1] ~ DiscreteTransition(s[1], A)
        for i in 2:3
            s[i] ~ DiscreteTransition(s[i - 1], B)
            y[i] ~ DiscreteTransition(s[i], A)
        end
        s[3] ~ Categorical(goal)
    end

    pred_model_constraints = @constraints begin
        q(s, B) = q(s)q(B)
    end

    @initialization function pred_model_init(q_B)
        q(B) = q_B
    end

    result = infer(
        model = pred_model(A = diageye(4), goal = [0, 1, 0, 0], p_B = DirichletCollection(ones(4, 4)), p_s_t = Categorical([0.7, 0.3, 0, 0])),
        data = (y = [[1, 0, 0, 0], missing, missing],),
        initialization = pred_model_init(DirichletCollection(ones(4, 4))),
        constraints = pred_model_constraints,
        iterations = 10
    )
    @test last(result.predictions[:y])[1] == Categorical([0.25, 0.25, 0.25, 0.25])

    pred_model_constraints = @constraints begin
        q(s, B) = q(s)q(B)
        q(y[1], s) = q(y[1])q(s)
    end
    result = infer(
        model = pred_model(A = diageye(4), goal = [0, 0, 1, 0], p_B = DirichletCollection(ones(4, 4)), p_s_t = Categorical([0.7, 0.3, 0, 0])),
        data = (y = UnfactorizedData([[1, 0, 0, 0], missing, missing]),),
        initialization = pred_model_init(DirichletCollection(ones(4, 4))),
        constraints = pred_model_constraints,
        iterations = 10
    )
    @test probvec(last(last(result.predictions[:y]))) ≈ [0, 0, 1, 0]
end

@testitem "Test inference benchmark statistics" begin
    using RxInfer

    callbacks = RxInferBenchmarkCallbacks()

    # A simple model for testing that resembles a simple kalman filter with
    # random walk state transition and unknown observational noise
    @model function test_model1(y)
        τ ~ Gamma(shape = 1.0, rate = 1.0)

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

    infer(model = test_model1(), data = (y = [1.0, 2.0, 3.0],), callbacks = callbacks, iterations = 10, initialization = init, constraints = test_model1_constraints())
    @test length(callbacks.before_model_creation_ts) == 1
    @test length(callbacks.after_model_creation_ts) == 1
    @test first(callbacks.before_model_creation_ts) < first(callbacks.after_model_creation_ts)
    @test length(callbacks.before_inference_ts) == 1
    @test length(callbacks.after_inference_ts) == 1
    @test first(callbacks.before_inference_ts) < first(callbacks.after_inference_ts)
    @test length(callbacks.before_iteration_ts) == 1
    @test length(callbacks.after_iteration_ts) == 1
    @test length(last(callbacks.before_iteration_ts)) == 10
    @test length(last(callbacks.after_iteration_ts)) == 10

    callbacks = RxInferBenchmarkCallbacks()
    for i in 1:10
        infer(model = test_model1(), data = (y = [1.0, 2.0, 3.0],), callbacks = callbacks, iterations = 10, initialization = init, constraints = test_model1_constraints())
        @test length(callbacks.before_model_creation_ts) == i
        @test length(callbacks.after_model_creation_ts) == i
        @test last(callbacks.before_model_creation_ts) < last(callbacks.after_model_creation_ts)
        @test length(callbacks.before_inference_ts) == i
        @test length(callbacks.after_inference_ts) == i
        @test last(callbacks.before_inference_ts) < last(callbacks.after_inference_ts)
        @test length(callbacks.before_iteration_ts) == i
        @test length(callbacks.after_iteration_ts) == i
        length(last(callbacks.before_iteration_ts)) == 10
        @test length(last(callbacks.after_iteration_ts)) == 10
    end

    @model function kalman_filter(x_prev_mean, x_prev_var, τ_shape, τ_rate, y)
        x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
        τ ~ Gamma(shape = τ_shape, rate = τ_rate)

        # Random walk with fixed precision
        x_current ~ Normal(mean = x_prev, precision = 1.0)
        y ~ Normal(mean = x_current, precision = τ)
    end

    # We assume the following factorisation between variables 
    # in the variational distribution
    @constraints function filter_constraints()
        q(x_prev, x_current, τ) = q(x_prev, x_current)q(τ)
    end
    static_observations = rand(300)
    callbacks           = RxInferBenchmarkCallbacks()
    datastream          = from(static_observations) |> map(NamedTuple{(:y,), Tuple{Float64}}, (d) -> (y = d,))
    autoupdates         = @autoupdates begin
        x_prev_mean, x_prev_var = mean_var(q(x_current))
        τ_shape = shape(q(τ))
        τ_rate = rate(q(τ))
    end

    init = @initialization begin
        q(x_current) = NormalMeanVariance(0.0, 1e3)
        q(τ) = GammaShapeRate(1.0, 1.0)
    end

    engine = infer(
        model          = kalman_filter(),
        constraints    = filter_constraints(),
        datastream     = datastream,
        autoupdates    = autoupdates,
        returnvars     = (:x_current,),
        keephistory    = 10_000,
        historyvars    = (x_current = KeepLast(), τ = KeepLast()),
        initialization = init,
        iterations     = 10,
        free_energy    = true,
        autostart      = true,
        callbacks      = callbacks
    )

    @test length(callbacks.before_model_creation_ts) == 1
    @test length(callbacks.after_model_creation_ts) == 1
    @test length(callbacks.before_autostart_ts) == 1
    @test length(callbacks.after_autostart_ts) == 1
end
