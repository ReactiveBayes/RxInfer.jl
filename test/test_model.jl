module RxInferModelTest

using Test
using RxInfer
using Random

@testset "@model macro tests" begin
    @testset "Tuple based variables usage #1" begin
        @model function mixture_model()
            mean1 ~ Normal(mean = 10, variance = 10000)
            mean2 ~ Normal(mean = -10, variance = 10000)
            prec1 ~ Gamma(shape = 1, rate = 1)
            prec2 ~ Gamma(shape = 1, rate = 1)

            selector ~ Bernoulli(0.3)
            mixture  ~ NormalMixture(selector, (mean1, mean2), (prec1, prec2))

            y = datavar(Float64)
            y ~ Normal(mean = mixture, variance = 1.0)
        end

        model, _ = create_model(mixture_model(), constraints = MeanField())

        @test model[:selector] isa RandomVariable
        @test model[:mixture] isa RandomVariable
        @test length(getnodes(model)) === 7
    end

    @testset "Broadcasting #1" begin
        @model function bsyntax1(n, broadcasting)
            m ~ NormalMeanPrecision(0.0, 1.0)
            t ~ Gamma(1.0, 1.0)
            y = datavar(Float64, n)

            nodes = Vector{Any}(undef, n)

            if broadcasting
                nodes, y .~ NormalMeanPrecision(m, t)
            else
                for i in 1:n
                    nodes[i], y[i] ~ NormalMeanPrecision(m, t)
                end
            end

            return nodes, y, m, t
        end

        n = 10
        # Test that both model create without any issues
        modelb, (nodesb, yb, mb, tb) = create_model(bsyntax1(n, true))
        model, (nodes, y, m, t)      = create_model(bsyntax1(n, false))

        # Test that degrees match
        @test ReactiveMP.degree(mb) === n + 1
        @test ReactiveMP.degree(m) === n + 1
        @test ReactiveMP.degree(tb) === n + 1
        @test ReactiveMP.degree(t) === n + 1

        for testset in (((nodesb, yb), mb, tb), ((nodes, y), m, t))
            for (node, y) in zip(testset[1]...)
                @test ReactiveMP.connectedvar(ReactiveMP.getinterface(node, :out)) === y # Test that :out interface has been connected to `y[i]`
                @test ReactiveMP.connectedvar(ReactiveMP.getinterface(node, :μ)) === testset[2] # Test that :μ interface has been connected to 'm'
                @test ReactiveMP.connectedvar(ReactiveMP.getinterface(node, :τ)) === testset[3] # Test that :τ interface has been connected to 't'
            end
        end
    end

    @testset "Broadcasting #2" begin
        import LinearAlgebra: det

        @model function bsyntax2(n)
            x = randomvar(n)
            y = datavar(Float64, n)

            nodes = Vector{Any}(undef, n)

            x[1] ~ NormalMeanVariance(0.0, 1.0)
            x[2:end] .~ x[1:(end - 1)] + 1
            nodes, y .~ NormalMeanVariance(x .+ 1 .- constvar(1) .+ 1, det((diageye(2) .+ diageye(2)) ./ 2)) where {q = q(μ)q(v)q(out), meta = 1}

            return (nodes,)
        end

        n = 10
        # Test that model creates without any issues
        model, (nodes,) = create_model(bsyntax2(n))

        @test length(getrandom(model)) === n + n + n + n

        for node in nodes
            v = ReactiveMP.connectedvar(ReactiveMP.getinterface(node, :v))
            @test ReactiveMP.factorisation(node) === ((1,), (2,), (3,))
            @test ReactiveMP.metadata(node) === 1
            @test v isa ReactiveMP.ConstVariable
            @test ReactiveMP.getconst(v) == 1
        end
    end

    @testset "Priors in arguments" begin
        @model function coin_model_priors1(n, prior)
            y = datavar(Float64, n)
            θ ~ prior
            for i in 1:n
                y[i] ~ Bernoulli(θ)
            end
        end

        @model function coin_model_priors2(n, prior)
            y = datavar(Float64, n)
            θ = randomvar()
            θ ~ prior
            for i in 1:n
                y[i] ~ Bernoulli(θ)
            end
        end

        @model function coin_model_priors3(n, priors)
            y = datavar(Float64, n)
            θ = randomvar(1)
            θ .~ priors
            for i in 1:n
                y[i] ~ Bernoulli(θ[1])
            end
        end

        rng  = MersenneTwister(42)
        n    = 50
        data = float.(rand(rng, Bernoulli(0.75), n))

        testsets = [(prior = Beta(4.0, 8.0), answer = Beta(43.0, 19.0)), (prior = Beta(54.0, 1.0), answer = Beta(93.0, 12.0)), (prior = Beta(1.0, 12.0), answer = Beta(40.0, 23.0))]

        for ts in testsets
            @test inference(model = coin_model_priors1(n, ts[:prior]), data = (y = data,)).posteriors[:θ] == ts[:answer]
            @test inference(model = coin_model_priors2(n, ts[:prior]), data = (y = data,)).posteriors[:θ] == ts[:answer]
            @test inference(model = coin_model_priors3(n, [ts[:prior]]), data = (y = data,)).posteriors[:θ] == [ts[:answer]]
        end
    end

    @testset "Error #1: Fail if variable in broadcasting hasnt been defined" begin
        @model function berror1(n)
            x = randomvar(n)
            x .~ NormalMeanVariance(0.0, 1.0)
            y .~ NormalMeanVariance(x, 1.0) # <- y has not be defined, but used in broadcasting
        end

        @test_throws ErrorException create_model(berror1(10))
    end

    @testset "Error #2: Fail if variables has been overwritten" begin
        @model function mymodel1(; condition)
            if condition === 0
                x = randomvar()
                x = randomvar()
                x ~ NormalMeanPrecision(0.0, 1.0)
            elseif condition === 1
                x ~ NormalMeanPrecision(0.0, 1.0)
                x = randomvar()
            elseif condition === 2
                x = randomvar()
                x_saved = x
                x ~ NormalMeanPrecision(0.0, 1.0)
                @test x_saved === x
            elseif condition === 3
                x ~ NormalMeanPrecision(0.0, 1.0)
            end

            y = datavar(Float64)
            y ~ NormalMeanPrecision(x, 1.0)
        end

        @test_throws UndefVarError create_model(mymodel1(condition = -1))
        @test_throws ErrorException create_model(mymodel1(condition = 0))
        @test_throws ErrorException create_model(mymodel1(condition = 1))

        m, _ = create_model(mymodel1(condition = 2))
        @test haskey(m, :x) && ReactiveMP.degree(m[:x]) === 2

        m, _ = create_model(mymodel1(condition = 3))
        @test haskey(m, :x) && ReactiveMP.degree(m[:x]) === 2
    end

    @testset "Error #3: make_node should throws on an unknown distribution type" begin
        struct DummyDistributionTestModelError3 <: Distribution{Univariate, Continuous} end

        @test_throws ErrorException ReactiveMP.make_node(FactorGraphModel(), FactorNodeCreationOptions(), DummyDistributionTestModelError3, AutoVar(:θ))

        @test_throws ErrorException ReactiveMP.make_node(FactorGraphModel(), FactorNodeCreationOptions(), DummyDistributionTestModelError3, randomvar(:θ))
    end
    @testset "Warning for unused datavars" begin
        @model function test_model1(n)
            x = randomvar(n)
            y = datavar(Float64, n)
        
        
            τ ~ Gamma(1.0, 1.0)
        
            x[1] ~ Normal(mean = 0.0, variance = 1.0)
            y[1] ~ Normal(mean = x[1], precision = τ)
        
            for i in 2:n-1
                x[i] ~ Normal(mean = x[i - 1], variance = 1.0)
                y[i] ~ Normal(mean = x[i], precision = τ)
            end
            # y_n is unused
            x[n] ~ Normal(mean = x[n - 1], variance = 1.0)
            y[n-1] ~ Normal(mean = x[n], precision = τ)
        
        end
        
        @constraints function test_model1_constraints()
            q(x, τ) = q(x)q(τ)
        end
        
        @constraints function test_model1_constraints()
            q(x, τ) = q(x)q(τ)
        end
        observations = rand(10)
    
        @test_logs (:warn, r"Unused data variable .*") result = inference(
            model = test_model1(10),
            constraints = test_model1_constraints(),
            data = (y = observations,),
            initmarginals = (τ = Gamma(1.0, 1.0),),
            iterations = 10,
            returnvars = KeepEach(),
            free_energy = true,
            warn=true,
        )
        @test_logs result = inference(
            model = test_model1(10),
            constraints = test_model1_constraints(),
            data = (y = observations,),
            initmarginals = (τ = Gamma(1.0, 1.0),),
            iterations = 10,
            returnvars = KeepEach(),
            free_energy = true,
            warn=false,
        )
    end
end

end
