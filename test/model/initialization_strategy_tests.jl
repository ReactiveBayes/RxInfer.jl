# Initialization strategies: `FromPriors` initialises every random variable an `@initialization`
# block leaves out from its prior, at its parents' initial means.

@testmodule InitializationStrategies begin
    using RxInfer
    import RxInfer: GraphPPL, InitMarExtraKey

    # The initial marginals of a model built with the initialization `initialization`, by name
    function initial_marginals(generator, data, initialization)
        plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin(), RxInfer.InitializationPlugin(initialization))
        model = RxInfer.getmodel(RxInfer.create_model(GraphPPL.with_plugins(generator, plugins) | data))
        marginals = Dict{Tuple{Symbol, Any}, Any}()
        GraphPPL.variable_nodes(model) do label, nodedata
            properties = GraphPPL.getproperties(nodedata)
            if GraphPPL.hasextra(nodedata, InitMarExtraKey)
                marginals[(GraphPPL.getname(properties), GraphPPL.index(properties))] = GraphPPL.getextra(nodedata, InitMarExtraKey)
            end
        end
        return marginals
    end

    @model function hierarchy(y)
        τ ~ Gamma(shape = 2.0, rate = 1.0)
        μ ~ Normal(mean = 1.0, precision = τ)
        for i in eachindex(y)
            y[i] ~ Normal(mean = μ, precision = τ)
        end
    end

    @model function nonlinear_chain(y)
        x[1] ~ Normal(mean = 0.5, variance = 1.0)
        for t in 2:length(y)
            z[t] := tanh(x[t - 1])
            x[t] ~ Normal(mean = z[t], variance = 0.1)
        end
        for t in eachindex(y)
            y[t] ~ Normal(mean = x[t], variance = 1.0)
        end
    end

    @model function coin(y)
        p ~ Beta(2.0, 3.0)
        y .~ Bernoulli(p)
    end

    # `x`'s parent is data, which has no value when the model is built
    @model function regression_on_data(y, u)
        x ~ Normal(mean = u, variance = 1.0)
        y ~ Normal(mean = x, variance = 1.0)
    end

    export initial_marginals, hierarchy, nonlinear_chain, coin, regression_on_data
end

@testitem "FromPriors: each variable starts from its prior at its parents' means" setup = [InitializationStrategies] begin
    using .InitializationStrategies
    marginals = initial_marginals(hierarchy(), (y = [1.0, 2.0],), FromPriors())
    @test marginals[(:τ, nothing)] == GammaShapeRate(2.0, 1.0)
    # the precision of μ is the initial mean of τ
    @test marginals[(:μ, nothing)] == NormalMeanPrecision(1.0, 2.0)

    marginals = initial_marginals(coin(), (y = [1.0, 0.0],), FromPriors())
    @test marginals[(:p, nothing)] == Beta(2.0, 3.0)
    @test length(marginals) == 1

    # down a chain, through a deterministic node that is not initialised itself
    marginals = initial_marginals(nonlinear_chain(), (y = [0.1, 0.2, 0.3],), FromPriors())
    @test marginals[(:x, 1)] == NormalMeanVariance(0.5, 1.0)
    @test marginals[(:x, 2)] == NormalMeanVariance(tanh(0.5), 0.1)
    @test marginals[(:x, 3)] == NormalMeanVariance(tanh(tanh(0.5)), 0.1)
    @test !any(key -> first(key) === :z, keys(marginals))
end

@testitem "FromPriors: a specification comes first and is never overwritten" setup = [InitializationStrategies] begin
    using .InitializationStrategies
    given = @initialization begin
        q(τ) = GammaShapeRate(5.0, 1.0)
    end
    marginals = initial_marginals(hierarchy(), (y = [1.0, 2.0],), FromPriors(given))
    @test marginals[(:τ, nothing)] == GammaShapeRate(5.0, 1.0)
    # its children see the given marginal
    @test marginals[(:μ, nothing)] == NormalMeanPrecision(1.0, 5.0)
    # a block alone initialises what it names
    @test initial_marginals(hierarchy(), (y = [1.0, 2.0],), given) == Dict((:τ, nothing) => GammaShapeRate(5.0, 1.0))
end

@testitem "FromPriors: a variable without a prior that can be formed is left out" setup = [InitializationStrategies] begin
    using .InitializationStrategies
    marginals = initial_marginals(regression_on_data(), (y = 1.0, u = 0.0), FromPriors())
    @test !haskey(marginals, (:x, nothing))
end

@testitem "FromPriors: inference without an initialization" setup = [InitializationStrategies] begin
    using .InitializationStrategies
    y = [0.8, 1.3, 1.1, 0.6]
    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end
    from_priors = infer(model = hierarchy(), data = (y = y,), constraints = constraints, initialization = FromPriors(), iterations = 10)
    # the same as initialising τ by hand with its prior, which mean field needs
    by_hand = infer(
        model = hierarchy(), data = (y = y,), constraints = constraints, iterations = 10,
        initialization = @initialization(
            begin
                q(τ) = GammaShapeRate(2.0, 1.0)
            end
        ),
    )
    @test mean(last(from_priors.posteriors[:μ])) ≈ mean(last(by_hand.posteriors[:μ]))
    @test mean(last(from_priors.posteriors[:τ])) ≈ mean(last(by_hand.posteriors[:τ]))
    @test repr(FromPriors()) == "FromPriors()"
end
