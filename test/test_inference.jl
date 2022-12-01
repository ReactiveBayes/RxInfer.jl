module RxInferInferenceTest

using Test
using RxInfer
using Random

@testset "__inference_check_itertype" begin
    import RxInfer: __inference_check_itertype

    @test __inference_check_itertype(:something, nothing) === nothing
    @test __inference_check_itertype(:something, (1,)) === nothing
    @test __inference_check_itertype(:something, (1, 2)) === nothing
    @test __inference_check_itertype(:something, []) === nothing
    @test __inference_check_itertype(:something, [1, 2]) === nothing

    @test_throws ErrorException __inference_check_itertype(:something, 1)
    @test_throws ErrorException __inference_check_itertype(:something, (1))
    @test_throws ErrorException __inference_check_itertype(:something, missing)
end

@testset "__inference_check_dicttype" begin
    import RxInfer: __inference_check_dicttype

    @test __inference_check_dicttype(:something, nothing) === nothing
    @test __inference_check_dicttype(:something, (x = 1,)) === nothing
    @test __inference_check_dicttype(:something, (x = 1, y = 2)) === nothing
    @test __inference_check_dicttype(:something, Dict(:x => 1)) === nothing
    @test __inference_check_dicttype(:something, Dict(:x => 1, :y => 2)) === nothing

    @test_throws ErrorException __inference_check_dicttype(:something, 1)
    @test_throws ErrorException __inference_check_dicttype(:something, (1))
    @test_throws ErrorException __inference_check_dicttype(:something, missing)
    @test_throws ErrorException __inference_check_dicttype(:something, (missing))
end

@testset "`@autoupdates` macro" begin
    function somefunction(something)
        return nothing
    end

    @testset "Use case #1" begin
        autoupdate = @autoupdates begin
            x_t_prev_mean = somefunction(q(x_t_current))
        end

        @test autoupdate isa Tuple && length(autoupdate) === 1
        @test autoupdate[1] isa RxInfer.RxInferenceAutoUpdateSpecification
        @test autoupdate[1].labels === (:x_t_prev_mean,)
        @test autoupdate[1].callback === somefunction
        @test autoupdate[1].from === RxInfer.FromMarginalAutoUpdate()
        @test autoupdate[1].variable === :x_t_current
    end

    @testset "Use case #2" begin
        autoupdate = @autoupdates begin
            x_t_prev_mean = somefunction(μ(x_t_current))
        end

        @test autoupdate isa Tuple && length(autoupdate) === 1
        @test autoupdate[1] isa RxInfer.RxInferenceAutoUpdateSpecification
        @test autoupdate[1].labels === (:x_t_prev_mean,)
        @test autoupdate[1].callback === somefunction
        @test autoupdate[1].from === RxInfer.FromMessageAutoUpdate()
        @test autoupdate[1].variable === :x_t_current
    end

    @testset "Use case #3" begin
        autoupdate = @autoupdates begin
            x_t_prev_mean, x_t_prev_var = somefunction(q(x_t_current))
            and_another_one = somefunction(μ(τ_current))
        end

        @test autoupdate isa Tuple && length(autoupdate) === 2
        @test autoupdate[1] isa RxInfer.RxInferenceAutoUpdateSpecification
        @test autoupdate[1].labels === (:x_t_prev_mean, :x_t_prev_var)
        @test autoupdate[1].callback === somefunction
        @test autoupdate[1].from === RxInfer.FromMarginalAutoUpdate()
        @test autoupdate[1].variable === :x_t_current

        @test autoupdate[2] isa RxInfer.RxInferenceAutoUpdateSpecification
        @test autoupdate[2].labels === (:and_another_one,)
        @test autoupdate[2].callback === somefunction
        @test autoupdate[2].from === RxInfer.FromMessageAutoUpdate()
        @test autoupdate[2].variable === :τ_current
    end

    @testset "Use case #4.1: simple indexing" begin
        autoupdate = @autoupdates begin
            x_t_prev_mean = somefunction(q(x[1]))
        end

        @test autoupdate isa Tuple && length(autoupdate) === 1
        @test autoupdate[1] isa RxInfer.RxInferenceAutoUpdateSpecification
        @test autoupdate[1].labels === (:x_t_prev_mean,)
        @test autoupdate[1].callback === somefunction
        @test autoupdate[1].from === RxInfer.FromMarginalAutoUpdate()
        @test autoupdate[1].variable === RxInfer.RxInferenceAutoUpdateIndexedVariable(:x, (1,))
    end

    @testset "Use case #4.2: complex indexing" begin
        autoupdate = @autoupdates begin
            x_t_prev_mean = somefunction(q(x[1, 2, 3]))
        end

        @test autoupdate isa Tuple && length(autoupdate) === 1
        @test autoupdate[1] isa RxInfer.RxInferenceAutoUpdateSpecification
        @test autoupdate[1].labels === (:x_t_prev_mean,)
        @test autoupdate[1].callback === somefunction
        @test autoupdate[1].from === RxInfer.FromMarginalAutoUpdate()
        @test autoupdate[1].variable === RxInfer.RxInferenceAutoUpdateIndexedVariable(:x, (1, 2, 3))
    end

    @testset "Error cases" begin
        # No specs
        @test_throws LoadError eval(:(@autoupdates begin end))

        # No specs
        @test_throws LoadError eval(:(@autoupdates begin
            x = 1
        end))

        # Complex lhs
        @test_throws LoadError eval(:(@autoupdates begin
            x[1] = somefunction(q(x_t_current))
        end))

        @test_throws LoadError eval(:(@autoupdates begin
            x[1] = somefunction(μ(x_t_current))
        end))

        # Complex call expression
        @test_throws LoadError eval(:(@autoupdates begin
            x = somefunction(q(x_t_current, 3))
        end))

        @test_throws LoadError eval(:(@autoupdates begin
            x = somefunction(q(x_t_current), 3)
        end))

        @test_throws LoadError eval(:(@autoupdates begin
            x = somefunction(μ(x_t_current), 3)
        end))
    end
end

@testset "Reactive inference with `rxinference` for test model #1" begin

    # A simple model for testing that resembles a simple kalman filter with
    # random walk state transition and unknown observational noise
    @model function test_model1()

        # Reactive prior inputs for `x_t_min`
        x_t_min_mean = datavar(Float64)
        x_t_min_var  = datavar(Float64)

        x_t_min ~ Normal(mean = x_t_min_mean, variance = x_t_min_var)

        # Reactive prior inputs for `τ`
        τ_shape = datavar(Float64)
        τ_rate  = datavar(Float64)

        τ ~ Gamma(shape = τ_shape, rate = τ_rate)

        # State transition
        x_t ~ Normal(mean = x_t_min, precision = 1.0)

        # Observations
        y = datavar(Float64)
        y ~ Normal(mean = x_t, precision = τ)

        return 2, 3.0, "hello world" # test returnval
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
        prevx = nextx
    end

    @testset "Check basic usage" begin
        for keephistory in (0, 1, 2), iterations in (3, 4), free_energy in (true, Float64, false), returnvars in ((:x_t,), (:x_t, :τ)), historyvars in ((:x_t,), (:x_t, :τ))
            historyvars = keephistory > 0 ? NamedTuple{historyvars}(map(_ -> KeepEach(), historyvars)) : nothing

            engine = rxinference(
                model = test_model1(),
                constraints = MeanField(),
                data = (y = observedy,),
                returnvars = returnvars,
                historyvars = historyvars,
                keephistory = keephistory,
                initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
                iterations = iterations,
                free_energy = free_energy,
                autoupdates = autoupdates
            )

            # Test that the `.model` reference is correct
            @test length(getnodes(engine.model)) === 4
            @test length(getrandom(engine.model)) === 3
            @test length(getdata(engine.model)) === 5
            @test length(getconstant(engine.model)) === 1

            # Test that the `.returnval` reference is correct
            @test engine.returnval === (2, 3.0, "hello world")

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

        engine = rxinference(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
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

        @test typeof(callbacksdata[1][2]) <: Tuple{}                                              # before_model_creation
        @test typeof(callbacksdata[2][2]) <: Tuple{FactorGraphModel, Tuple{Int, Float64, String}} # after_model_creation 
        @test typeof(callbacksdata[3][2]) <: Tuple{RxInferenceEngine}                             # before_autostart 
        @test typeof(callbacksdata[4][2]) <: Tuple{RxInferenceEngine}                             # after_autostart
    end

    @testset "Check callbacks usage: autostart disabled" begin
        callbacksdata = []

        engine = rxinference(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
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

        @test typeof(callbacksdata[1][2]) <: Tuple{}                                              # before_model_creation
        @test typeof(callbacksdata[2][2]) <: Tuple{FactorGraphModel, Tuple{Int, Float64, String}} # after_model_creation 

        RxInfer.start(engine)

        # Nothing extra should has been executed on `start`
        @test length(callbacksdata) === 2
    end

    @testset "Check callbacks usage: unknown callback warning" begin
        callbacksdata = []

        @test_logs (:warn, r"Unknown callback specification.*hello_world.*Available callbacks.*") result = rxinference(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
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
            engine = rxinference(
                model = test_model1(),
                constraints = MeanField(),
                data = (y = observedy,),
                initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
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
                @test map(name, filter(event -> event isa RxInferenceEvent{:before_iteration} || event isa RxInferenceEvent{:after_iteration}, events)) ==
                    repeat([:before_iteration, :after_iteration], iterations)

                # Check that the number of `:before_auto_update` and `:after_auto_update` events depends on the number of iterations
                @test length(filter(event -> event isa RxInferenceEvent{:before_auto_update}, events)) == iterations
                @test length(filter(event -> event isa RxInferenceEvent{:after_auto_update}, events)) == iterations

                # Check the associated data with the `:before_auto_update` events
                foreach(enumerate(filter(event -> event isa RxInferenceEvent{:before_auto_update}, events))) do (ii, event)
                    model, iteration, fupdate = event
                    @test model === engine.model
                    @test iteration === ii
                    @test length(fupdate) === 3
                    @test name.(getindex.(Iterators.flatten(collect.(fupdate)), 1)) == [:x_t_min_mean, :x_t_min_var, :τ_shape, :τ_rate]
                    @test eltype(getindex.(Iterators.flatten(collect.(fupdate)), 2)) === Float64
                end

                # Check the associated data with the `:after_auto_update` events
                foreach(enumerate(filter(event -> event isa RxInferenceEvent{:after_auto_update}, events))) do (ii, event)
                    model, iteration, fupdate = event
                    @test model === engine.model
                    @test iteration === ii
                    @test length(fupdate) === 3
                    @test name.(getindex.(Iterators.flatten(collect.(fupdate)), 1)) == [:x_t_min_mean, :x_t_min_var, :τ_shape, :τ_rate]
                    @test eltype(getindex.(Iterators.flatten(collect.(fupdate)), 2)) === Float64
                end

                # Check the correct ordering of the `:before_auto_update` and `:after_auto_update` events
                @test map(name, filter(event -> event isa RxInferenceEvent{:before_auto_update} || event isa RxInferenceEvent{:after_auto_update}, events)) ==
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
                @test map(name, filter(event -> event isa RxInferenceEvent{:before_data_update} || event isa RxInferenceEvent{:after_data_update}, events)) ==
                    repeat([:before_data_update, :after_data_update], iterations)

                # Check the correct ordering of the iteration related events
                @test map(
                    name,
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

    @testset "Check the event creation and unrolling syntax" begin
        data1, data2 = RxInferenceEvent(Val(:event1), (1, 2.0))

        @test data1 === 1
        @test data2 === 2.0
    end

    @testset "Either `data` or `datastream` is required" begin
        @test_throws ErrorException rxinference(model = test_model1())
    end

    @testset "`data` and `datastream` cannot be used together" begin
        @test_throws ErrorException rxinference(model = test_model1(), data = (y = observedy,), datastream = labeled(Val((:y,)), combineLatest(from(observedy))))
    end
end

end
