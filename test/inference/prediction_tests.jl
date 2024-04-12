@testitem "Predictions in IID Beta-Bernoulli" begin
    @model function beta_bernoulli_single(y, a, b)
        θ ~ Beta(a, b)
        y ~ Bernoulli(θ)
    end

    @model function beta_bernoulli_multiple(y, n, a, b)
        θ ~ Beta(a, b)
        for i in 1:n
            y[i] ~ Bernoulli(θ)
        end
    end

    # The model is simple enough to be able to calculate the expected prediction
    # We check two models beta_bernoulli_single and beta_bernoulli_multiple
    # They are essentially the same with the exception that the second model has multiple observations
    # Each case has two options: KeepLast() and KeepEach() which specifies how many predictions we should keep for the result
    for a in (1, 2, 3), b in (2, 3, 4), iterations in (10, 20, 30)
        expected_prediction = @call_rule Bernoulli(:out, Marginalisation) (q_p = Beta(a, b),)

        @testset "beta_bernoulli_single" for option in (KeepLast(), KeepEach())
            result_missing_data_explicitly_specified_1 = infer(
                model = beta_bernoulli_single(a = a, b = b), data = (y = missing,), predictvars = (y = option,), iterations = iterations
            )
            result_missing_data_explicitly_specified_2 = infer(model = beta_bernoulli_single(a = a, b = b), data = (y = missing,), predictvars = option, iterations = iterations)
            result_data_is_not_explicitly_specified = infer(model = beta_bernoulli_single(a = a, b = b), predictvars = (y = option,), iterations = iterations)

            results = [result_missing_data_explicitly_specified_1, result_missing_data_explicitly_specified_2, result_data_is_not_explicitly_specified]

            for result in results
                if option === KeepLast()
                    # We should get a single prediction (keep the last one)
                    @test result.predictions[:y] === expected_prediction
                    # The posteriors however are not explicitly specified, should be derived from the number of iterations
                    @test length(result.posteriors[:θ]) === iterations
                    # Sine the data is missing the posteriors should be simply equal to the prior
                    @test all(result.posteriors[:θ] .== Beta(a, b))
                elseif option === KeepEach()
                    # We should get as many predictions as number of iterations (keep each )
                    @test length(result.predictions[:y]) === iterations
                    @test all(result.predictions[:y] .== expected_prediction)
                    # The posteriors however are not explicitly specified, should be derived from the number of iterations
                    @test length(result.posteriors[:θ]) === iterations
                    # Sine the data is missing the posteriors should be simply equal to the prior
                    @test all(result.posteriors[:θ] .== Beta(a, b))
                end
            end
        end

        @testset "beta_bernoulli_multiple" begin
            for n in (10, 20, 30), option in (KeepLast(), KeepEach())
                expected_prediction = @call_rule Bernoulli(:out, Marginalisation) (q_p = Beta(a, b),)

                result_missing_data_explicitly_specified_1 = infer(
                    model = beta_bernoulli_multiple(a = a, b = b, n = n), data = (y = missing,), predictvars = (y = option,), iterations = iterations
                )
                result_missing_data_explicitly_specified_2 = infer(
                    model = beta_bernoulli_multiple(a = a, b = b, n = n), data = (y = missing,), predictvars = option, iterations = iterations
                )
                result_data_is_not_explicitly_specified = infer(model = beta_bernoulli_multiple(a = a, b = b, n = n), predictvars = (y = option,), iterations = iterations)

                results = [result_missing_data_explicitly_specified_1, result_missing_data_explicitly_specified_2, result_data_is_not_explicitly_specified]

                for result in results
                    if option === KeepLast()
                        # We should get a single prediction for each element of `y` (keep the last one)
                        @test length(result.predictions[:y]) === n
                        @test all(result.predictions[:y] .=== expected_prediction)
                        # The posteriors however are not explicitly specified, should be derived from the number of iterations
                        @test length(result.posteriors[:θ]) === iterations
                        # Sine the data is missing the posteriors should be simply equal to the prior
                        @test all(result.posteriors[:θ] .== Beta(a, b))
                    elseif option === KeepEach()
                        # We should get as many predictions as number of iterations for each element of `y` (keep each)
                        @test length(result.predictions[:y]) === iterations
                        @test all(length.(result.predictions[:y]) .== n)
                        foreach(result.predictions[:y]) do prediction
                            @test all(prediction .== expected_prediction)
                        end
                        # The posteriors however are not explicitly specified, should be derived from the number of iterations
                        @test length(result.posteriors[:θ]) === iterations
                        # Sine the data is missing the posteriors should be simply equal to the prior
                        @test all(result.posteriors[:θ] .== Beta(a, b))
                    end
                end
            end
        end
    end

    @testset "Predict vars is specified implicitly without data provided, should error" begin
        @test_throws "Make sure to provide `data` or specify `predictvars` explicitly." infer(model = beta_bernoulli_single(a = 1, b = 1), predictvars = KeepEach())
        @test_throws "Make sure to provide `data` or specify `predictvars` explicitly." infer(model = beta_bernoulli_single(a = 1, b = 1), predictvars = KeepLast())
        @test_throws "Make sure to provide `data` or specify `predictvars` explicitly." infer(model = beta_bernoulli_multiple(a = 1, b = 1, n = 10), predictvars = KeepEach())
        @test_throws "Make sure to provide `data` or specify `predictvars` explicitly." infer(model = beta_bernoulli_multiple(a = 1, b = 1, n = 10), predictvars = KeepLast())
    end
end

@testitem "Single prediction, no data provided" begin
    @model function simple_model(y)
        z ~ NormalMeanPrecision(0, 1.0)
        x ~ NormalMeanPrecision(z, 1.0)
        y ~ NormalMeanPrecision(x, 10.0)
    end

    result = infer(model = simple_model(), data = (y = missing,), predictvars = KeepLast())

    @test haskey(result.predictions, :y)
    @test typeof(result.predictions[:y]) <: NormalDistributionsFamily

    # no comma after `missing` should throw an error
    @test_throws "Keyword argument `data` expects either `Dict` or `NamedTuple` as an input" infer(model = simple_model(), data = (y = missing), predictvars = KeepLast())
end

@testitem "Single prediction, data is provided" begin
    @model function simple_model(y, x)
        a ~ Normal(mean = x, var = 1.0)
        b ~ Normal(mean = a, var = 1.0)
        c ~ Normal(mean = b, var = 1.0)
        d ~ Normal(mean = c, var = 1.0)
        y ~ Normal(mean = d, var = 1.0)
    end

    result = infer(model = simple_model(), data = (y = missing, x = 1.0))

    @test haskey(result.predictions, :y)
    @test !haskey(result.predictions, :x)
    @test typeof(result.predictions[:y]) <: NormalDistributionsFamily

    result = infer(model = simple_model(), data = (y = missing, x = 1.0), predictvars = (x = KeepLast(),))

    @test haskey(result.predictions, :y)
    @test haskey(result.predictions, :x)
    @test typeof(result.predictions[:y]) <: NormalDistributionsFamily
    @test typeof(result.predictions[:x]) <: NormalDistributionsFamily
end

@testitem "Predictions in State Space Models" begin

    # A simple model for testing that resembles a simple kalman smoother with
    # random walk state transition
    @model function model_1(y, o)
        n = length(y)

        x_0 ~ NormalMeanPrecision(0.0, 1.0)

        z = x_0
        for i in 1:n
            x[i] ~ NormalMeanPrecision(z, 1.0)
            y[i] ~ NormalMeanPrecision(x[i], 1.0)
            z = x[i]
        end
        x[n + 1] ~ NormalMeanPrecision(z, 1.0)
        o[1] ~ NormalMeanPrecision(x[n + 1], 1.0)
        z = x[n + 1]
        x[n + 2] ~ NormalMeanPrecision(x[n + 1], 1.0)
        o[2] ~ NormalMeanPrecision(x[n + 2], 1.0)
    end

    @testset "test #1 (array with missing + KeepLast predictvars)" begin
        for data in [(y = [1.0, -500.0, missing, 100.0],), (y = [1.0, -500.0, missing, 100.0, missing, missing],)], iterations in (10, 20)
            result = infer(model = model_1(), iterations = iterations, data = data, predictvars = (o = KeepLast(),))

            @test haskey(result.predictions, :o)
            @test haskey(result.predictions, :y)

            # The prediction for `o` should have the same length as the `o` handler 
            # since we use the `KeepLast` strategy
            @test length(result.predictions[:o]) === 2
            @test all(typeof.(result.predictions[:o]) .<: NormalDistributionsFamily)

            # Here the predictions for `y` should have the same length as the number of iterations
            @test length(result.predictions[:y]) === iterations
            foreach(result.predictions[:y]) do prediction
                # And each individual prediction should of the same length as the data
                @test length(prediction) === length(data[:y])
                foreach(prediction) do ŷᵢ
                    # And each individual element of the prediction must be of type NormalDistributionsFamily
                    @test typeof(ŷᵢ) <: NormalDistributionsFamily
                end
            end
        end
    end

    @testset "test #1.1 (KeepEach predictvars)" begin
        for data in [(y = [1.0, -500.0, 1.0, 100.0],), (y = [1.0, -500.0, 3.0, 100.0, 4.0, 5.0],)], iterations in (10, 20)
            result = infer(model = model_1(), iterations = iterations, data = data, predictvars = (o = KeepEach(),))

            @test haskey(result.predictions, :o)
            @test !haskey(result.predictions, :y)

            # Here the predictions for `o` should have the same length as the number of iterations
            @test length(result.predictions[:o]) === iterations
            foreach(result.predictions[:o]) do prediction
                # And each individual prediction should of the `o` handle inside the model (which is equal to `2`)
                @test length(prediction) === 2
                foreach(prediction) do ôᵢ
                    # And each individual element of the prediction must be of type NormalDistributionsFamily
                    @test typeof(ôᵢ) <: NormalDistributionsFamily
                end
            end
        end
    end

    @model function model_2(y, o)
        n = length(y)
        local x
        z ~ NormalMeanPrecision(0.0, 100.0)
        for i in 1:n
            x[i] ~ NormalMeanPrecision(z, 1.0)
            y[i] ~ NormalMeanPrecision(x[i], 1.0)
        end
        x[n + 1] ~ NormalMeanPrecision(x[n], 1.0)
        o ~ NormalMeanPrecision(x[n + 1], 1.0)
    end

    @testset "test #2 (array with missing + single entry for predictvars)" begin
        data = (y = [1.0, -10.0, 0.9, missing, missing],)

        for iterations in (10, 20)
            result = infer(model = model_2(), iterations = iterations, data = data, predictvars = (o = KeepEach(), y = KeepLast()))

            # note we used KeepEach for variable o with BP algorithm (10 iterations), 
            # we expect all predicted variables to be equal (because of the beleif propagation)
            @test all(prediction -> prediction == result.predictions[:o][1], result.predictions[:o])

            # predictions for `y` are saved for the last iteration
            # and should be of type NormalDistributionsFamily
            @test length(result.predictions[:o]) === iterations
            @test all(typeof.(result.predictions[:y]) .<: NormalDistributionsFamily)
        end
    end

    @model function model_3(y, o)
        n = length(y)

        local x
        z ~ NormalMeanPrecision(0.0, 100.0)

        for i in 1:n
            x[i] ~ NormalMeanPrecision(z, 1.0)
            y[i] ~ NormalMeanPrecision(x[i], 1.0)
        end
        x[n + 1] ~ NormalMeanPrecision(x[n], 1.0)
        o ~ NormalMeanPrecision(x[n + 1], 1.0)
    end

    @testset "test #3 (array + single entry for predictvars)" begin
        data = (y = [1.0, -10.0, 0.9],)
        result = infer(model = model_3(), iterations = 10, data = data, predictvars = (o = KeepLast(),))

        @test !haskey(result.predictions, :y)
        @test haskey(result.predictions, :o)
        @test typeof(result.predictions[:o]) <: NormalDistributionsFamily
    end

    @model function model_4(y)
        z ~ NormalMeanPrecision(3, 100.0)
        for i in eachindex(y)
            x[i] ~ NormalMeanPrecision(z, 1.0)
            y[i] ~ NormalMeanPrecision(x[i], 1.0)
        end
    end

    @testset "test #4 (array with a missing + no predictvars)" begin
        data = (y = [1.0, 2.0, missing],)
        result = infer(model = model_4(), iterations = 10, data = data)

        @test length(result.predictions[:y]) === 10
        foreach(result.predictions[:y]) do prediction
            @test length(prediction) === length(data[:y])
            @test all(typeof.(prediction) .<: NormalDistributionsFamily)
        end
    end

    @model function vmp_model(y, o)
        n = length(y)

        x_0 ~ NormalMeanPrecision(0.0, 100.0)
        γ ~ GammaShapeRate(1.0, 1.0)

        local x
        x_prev = x_0
        for i in 1:n
            x[i] ~ NormalMeanPrecision(x_prev, γ)
            y[i] ~ NormalMeanPrecision(x[i], 1.0)
            x_prev = x[i]
        end
        x[n + 1] ~ NormalMeanPrecision(x[n], 1.0)
        o ~ NormalMeanPrecision(x[n + 1], 1.0)
    end

    init = @initialization begin
        q(γ) = GammaShapeRate(1.0, 1.0)
    end

    @testset "test #5 vmp model" begin
        data = (y = [1.0, -10.0, 5.0],)

        constraints = @constraints begin
            q(x_0, x, γ) = q(x_0, x)q(γ)
        end

        result = infer(
            model = vmp_model(),
            data = data,
            constraints = constraints,
            free_energy = false,
            initialization = init,
            iterations = 10,
            returnvars = (γ = KeepEach(),),
            predictvars = (o = KeepEach(),)
        )

        @test haskey(result.predictions, :o)
        @test !haskey(result.predictions, :y)
        @test !haskey(result.predictions, :γ)

        @test !haskey(result.posteriors, :o)
        @test !haskey(result.posteriors, :y)
        @test haskey(result.posteriors, :γ)

        @test first(result.posteriors[:γ]) != last(result.posteriors[:γ])
        @test first(result.predictions[:o]) != last(result.predictions[:o])
        @test length(result.posteriors[:γ]) === 10
        @test length(result.predictions[:o]) === 10
    end
end
