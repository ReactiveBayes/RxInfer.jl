module RxInferInferenceTest

using Test
using RxInfer
using Random

@testset "__inference_check_itertype" begin 
    import RxInfer: __inference_check_itertype

    @test __inference_check_itertype(:something, nothing) === nothing
    @test __inference_check_itertype(:something, (1, )) === nothing
    @test __inference_check_itertype(:something, (1, 2)) === nothing
    @test __inference_check_itertype(:something, [ ]) === nothing
    @test __inference_check_itertype(:something, [ 1, 2 ]) === nothing

    @test_throws ErrorException __inference_check_itertype(:something, 1)
    @test_throws ErrorException __inference_check_itertype(:something, (1))
    @test_throws ErrorException __inference_check_itertype(:something, missing)
end

@testset "__inference_check_dicttype" begin 
    import RxInfer: __inference_check_dicttype

    @test __inference_check_dicttype(:something, nothing) === nothing
    @test __inference_check_dicttype(:something, (x = 1, )) === nothing
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
        @test autoupdate[1].labels === (:x_t_prev_mean, )
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
        @test autoupdate[1].labels === (:x_t_prev_mean, )
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
        @test autoupdate[2].labels === (:and_another_one, )
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

@testset "Reactive inference with `rxinference`" begin

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
        
    end

    @testset "Check basic usage on a simple model" begin

        n       = 100
        hiddenx   = [ ]
        observedy = [ ]
        prevx = 0.0
        for i in 1:n
            nextx = rand(NormalMeanVariance(prevx, 1.0))
            nexty = rand(NormalMeanPrecision(nextx, 10.0))
            push!(hiddenx, nextx)
            push!(observedy, nexty)
            prevx = nextx
        end

        datastream = from(observedy)

        result = rxinference(
            model       = kalman_filter(),
            constraints = MeanField(),
            data        = datastream,
            returnvars  = KeepEach(),
            initmarginals = (x_t = NormalMeanVariance(0.0, 1e3), τ = GammaShapeRate(1.0, 1.0)),
            iterations = 2,
            free_energy = true,
            redirect = (
                x_t = (q) -> (x_t_min_mean = mean(q), x_t_min_var = var(q)),
                τ = (q) -> (τ_shape = shape(q), τ_rate = rate(q))
            )
        );
    end

end

end