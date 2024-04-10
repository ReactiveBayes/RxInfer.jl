@testitem "Model can be conditioned on fixed data and materialized" begin
    import RxInfer: condition_on, create_model, getconditioned_on, DefferedDataHandler

    @model function beta_bernoulli(y, a, b)
        θ ~ Beta(a, b)
        y .~ Bernoulli(θ)
    end

    @testset "Conditioning on `y = [0.0, 1.0, 0.0]`" begin
        model_generator           = beta_bernoulli(a = 1.0, b = 1.0)
        model_generator_with_data = condition_on(model_generator, y = [0.0, 1.0, 0.0])

        @test getconditioned_on(model_generator_with_data) == (y = [0.0, 1.0, 0.0],)
        @test occursin("beta_bernoulli(a = 1.0, b = 1.0) conditioned on: ", repr(model_generator_with_data))
        @test occursin("y = [0.0, 1.0, 0.0]", repr(model_generator_with_data))
        @test create_model(model_generator_with_data) isa ProbabilisticModel
    end

    @testset "Conditioning on `y = [0.0, 1.0, 0.0], b = 7.0`" begin
        model_generator           = beta_bernoulli(a = 2.0)
        model_generator_with_data = condition_on(model_generator, y = [0.0, 1.0, 0.0], b = 7.0)

        @test getconditioned_on(model_generator_with_data) == (y = [0.0, 1.0, 0.0], b = 7.0)
        @test occursin("beta_bernoulli(a = 2.0) conditioned on: ", repr(model_generator_with_data))
        @test occursin("y = [0.0, 1.0, 0.0]", repr(model_generator_with_data))
        @test occursin("b = 7.0", repr(model_generator_with_data))
        @test create_model(model_generator_with_data) isa ProbabilisticModel
    end

    @model function beta_bernoulli_single(y, a, b)
        θ ~ Beta(a, b)
        y ~ Bernoulli(θ)
    end

    @testset "Conditioning on `y = ???`" begin
        model_generator           = beta_bernoulli_single(a = 1.0, b = 1.0)
        model_generator_with_data = condition_on(model_generator, y = DefferedDataHandler())

        @test getconditioned_on(model_generator_with_data) == (y = DefferedDataHandler(),)
        @test occursin("beta_bernoulli_single(a = 1.0, b = 1.0) conditioned on: ", repr(model_generator_with_data))
        @test occursin("y = [ deffered data ]", repr(model_generator_with_data))
        @test create_model(model_generator_with_data) isa ProbabilisticModel
    end
end

@testitem "We should be able to condition with the `|` operator" begin
    import RxInfer: create_model

    @model function beta_bernoulli(y, a, b)
        θ ~ Beta(a, b)
        y .~ Bernoulli(θ)
    end

    conditioned = beta_bernoulli(a = 1.0, b = 1.0) | (y = [1.0, 0.0, 3.0],)

    @test occursin("beta_bernoulli(a = 1.0, b = 1.0) conditioned on: ", repr(conditioned))
    @test occursin("y = [1.0, 0.0, 3.0]", repr(conditioned))
    @test create_model(conditioned) isa ProbabilisticModel
end

@testitem "create_deffered_data_handlers" begin
    import RxInfer: create_deffered_data_handlers, DefferedDataHandler

    @testset "Creating deffered labels from tuple of symbols" begin
        @test create_deffered_data_handlers((:x, :y)) === (x = DefferedDataHandler(), y = DefferedDataHandler())
    end

    @testset "Creating deffered labels from array of symbols" begin
        @test create_deffered_data_handlers([:x, :y]) == Dict(:x => DefferedDataHandler(), :y => DefferedDataHandler())
    end
end

@testitem "append_deffered_data_handlers" begin
    import RxInfer: append_deffered_data_handlers, DefferedDataHandler

    @testset "Append deffered data handlers to a named tuple from a tuple of symbols" begin
        @test append_deffered_data_handlers((z = 1,), (:x, :y)) == (z = 1, x = DefferedDataHandler(), y = DefferedDataHandler())
    end

    @testset "Append deffered data handlers to a Dict from a tuple of symbols" begin
        @test append_deffered_data_handlers(Dict(:z => 1), (:x, :y)) == Dict(:z => 1, :x => DefferedDataHandler(), :y => DefferedDataHandler())
    end

    @testset "Append deffered data handlers to a named tuple from a vector of symbols" begin
        @test append_deffered_data_handlers((z = 1,), [:x, :y]) == Dict(:z => 1, :x => DefferedDataHandler(), :y => DefferedDataHandler())
    end

    @testset "Append deffered data handlers to a Dict from a vector of symbols" begin
        @test append_deffered_data_handlers(Dict(:z => 1), [:x, :y]) == Dict(:z => 1, :x => DefferedDataHandler(), :y => DefferedDataHandler())
    end

    @testset "Conflicting names should throw a user-friendly errors" begin
        @test_throws "Cannot add `DefferedDataHandler` for the key `z`. Data has already been defined for the key `z`" append_deffered_data_handlers(Dict(:z => 1), [:z, :y])
        @test_throws "Cannot add `DefferedDataHandler` for the key `y`. Data has already been defined for the key `y`" append_deffered_data_handlers((y = 1,), (:y,))
    end
end