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