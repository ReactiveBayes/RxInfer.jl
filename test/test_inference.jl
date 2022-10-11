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

        # Complex rhs
        @test_throws LoadError eval(:(@autoupdates begin
            x = somefunction(q(x_t_current[1]))
        end))

        @test_throws LoadError eval(:(@autoupdates begin
            x = somefunction(μ(x_t_current[1]))
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

    n         = 100
    hiddenx   = []
    observedy = []
    prevx     = 0.0
    for i in 1:n
        nextx = rand(NormalMeanVariance(prevx, 1.0))
        nexty = rand(NormalMeanPrecision(nextx, 10.0))
        push!(hiddenx, nextx)
        push!(observedy, nexty)
        prevx = nextx
    end

    @testset "Check basic usage" begin
        
        for keephistory in (2, 3), iterations in (4, 5)
            result = rxinference(
                model = test_model1(),
                constraints = MeanField(),
                data = (y = observedy,),
                returnvars = (:x_t,),
                historyvars = (x_t = KeepEach(), τ = KeepEach()),
                keephistory = keephistory,
                initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
                iterations = iterations,
                free_energy = true,
                autoupdates = autoupdates,
            )

            # Test that the `.model` reference is correct
            @test length(getnodes(result.model)) === 4
            @test length(getrandom(result.model)) === 3
            @test length(getdata(result.model)) === 5
            @test length(getconstant(result.model)) === 1

            # Test that the `.returnval` reference is correct
            @test result.returnval === (2, 3.0, "hello world")

            @test sort(collect(keys(result.posteriors))) == [ :x_t ]
            @test sort(collect(keys(result.history))) == [ :x_t, :τ ]

            @test length(result.history[:x_t]) === keephistory 
            @test length(result.history[:x_t][end]) === iterations

            @test length(result.history[:τ]) === keephistory
            @test length(result.history[:τ][end]) === iterations

            @test length(result.free_energy_history) === iterations
            @test all(<=(0), diff(result.free_energy_history))

            @test length(result.free_energy_final_only_history) === keephistory
            @test length(result.free_energy_raw_history) === keephistory * iterations
        end
    end

    @testset "Check callbacks usage: autostart enabled" begin

        callbacksdata = []
        
        result = rxinference(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
            autoupdates = autoupdates,
            callbacks = (
                before_model_creation = (args...) -> push!(callbacksdata, (:before_model_creation, args)),
                after_model_creation = (args...) -> push!(callbacksdata, (:after_model_creation, args)),
                before_autostart = (args...) -> push!(callbacksdata, (:before_autostart, args)),
                after_autostart = (args...) -> push!(callbacksdata, (:after_autostart, args)),
            ),
            autostart = true
        )

        # First check the order
        @test first.(callbacksdata) == [
            :before_model_creation,
            :after_model_creation,
            :before_autostart,
            :after_autostart,
        ]

        @test typeof(callbacksdata[1][2]) <: Tuple{}                                              # before_model_creation
        @test typeof(callbacksdata[2][2]) <: Tuple{FactorGraphModel, Tuple{Int, Float64, String}} # after_model_creation 
        @test typeof(callbacksdata[3][2]) <: Tuple{RxInferenceEngine}                             # before_autostart 
        @test typeof(callbacksdata[4][2]) <: Tuple{RxInferenceEngine}                             # after_autostart

    end

    @testset "Check callbacks usage: autostart disabled" begin

        callbacksdata = []
        
        result = rxinference(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
            autoupdates = autoupdates,
            callbacks = (
                before_model_creation = (args...) -> push!(callbacksdata, (:before_model_creation, args)),
                after_model_creation = (args...) -> push!(callbacksdata, (:after_model_creation, args)),
                before_autostart = (args...) -> push!(callbacksdata, (:before_autostart, args)),
                after_autostart = (args...) -> push!(callbacksdata, (:after_autostart, args)),
            ),
            autostart = false
        )

        # First check the order
        @test first.(callbacksdata) == [
            :before_model_creation,
            :after_model_creation,
        ]

        @test typeof(callbacksdata[1][2]) <: Tuple{}                                              # before_model_creation
        @test typeof(callbacksdata[2][2]) <: Tuple{FactorGraphModel, Tuple{Int, Float64, String}} # after_model_creation 

        RxInfer.start(result)

        # Nothing extra should has been executed on `start`
        @test length(callbacksdata) === 2

    end

    @testset "Check callbacks usage: unknown callback warning" begin

        callbacksdata = []
        
        @test_warn r"Unknown callback specification.*hello_world.*Available callbacks.*" result = rxinference(
            model = test_model1(),
            constraints = MeanField(),
            data = (y = observedy,),
            initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
            autoupdates = autoupdates,
            callbacks = (
                hello_world = (args...) -> push!(callbacksdata, args),
            ),
            autostart = true
        )

        @test length(callbacksdata) === 0
    end

    @testset "Check the event creation and unrolling syntax" begin 
        data1, data2 = RxInferenceEvent(Val(:event1), (1, 2.0))

        @test data1 === 1
        @test data2 === 2.0
    end
end

end
