@testitem "Predictions in IID Beta-Bernoulli" begin
    @model function beta_bernoulli(y, a, b)
        θ ~ Beta(a, b)
        y ~ Bernoulli(θ)
    end

    @testset "Missing data, explicitly specified, keep last" for (a, b) in [(1, 1), (2, 3), (7, 8)]
        expected_prediction = @call_rule Bernoulli(:out, Marginalisation) (q_p = Beta(a, b), )
        result_keep_last = infer(model = beta_bernoulli(a = a, b = b), data = (y = missing,), predictvars = (y = KeepLast(),), iterations = 10)
        # We should get a single prediction (keep the last one)
        @test result_keep_last.predictions[:y] === expected_prediction
        # The posteriors however are not explicitly specified, should be derived from the number of iterations
        @test length(result_keep_last.posteriors[:θ]) === 10
        # Sine the data is missing the posteriors should be simply equal to the prior
        @test all(result_keep_last.posteriors[:θ] .== Beta(a, b))
    end

    @testset "Missing data, explicitly specified, keep last" for (a, b) in [(1, 1), (2, 3), (7, 8)]
        expected_prediction = @call_rule Bernoulli(:out, Marginalisation) (q_p = Beta(a, b), )
        result_keep_each = infer(model = beta_bernoulli(a = a, b = b), data = (y = missing,), predictvars = (y = KeepEach(),), iterations = 10)
        # We should get as many predictions as number of iterations (keep each )
        @test length(result_keep_each.predictions[:y]) === 10
        @test all(result_keep_each.predictions[:y] .== expected_prediction)
        # The posteriors however are not explicitly specified, should be derived from the number of iterations
        @test length(result_keep_each.posteriors[:θ]) === 10
        # Sine the data is missing the posteriors should be simply equal to the prior
        @test all(result_keep_each.posteriors[:θ] .== Beta(a, b))
    end

    @testset "Missing data, implicitly specified, keep last" for (a, b) in [(1, 1), (2, 3), (7, 8)]
        expected_prediction = @call_rule Bernoulli(:out, Marginalisation) (q_p = Beta(a, b), )
        result_keep_last = infer(model = beta_bernoulli(a = a, b = b), predictvars = (y = KeepLast(),), iterations = 10)
        # We should get a single prediction (keep the last one)
        @test result_keep_last.predictions[:y] === expected_prediction
        # The posteriors however are not explicitly specified, should be derived from the number of iterations
        @test length(result_keep_last.posteriors[:θ]) === 10
        # Sine the data is missing the posteriors should be simply equal to the prior
        @test all(result_keep_last.posteriors[:θ] .== Beta(a, b))
    end

    @testset "Missing data, implicitly specified, keep last" for (a, b) in [(1, 1), (2, 3), (7, 8)]
        expected_prediction = @call_rule Bernoulli(:out, Marginalisation) (q_p = Beta(a, b), )
        result_keep_each = infer(model = beta_bernoulli(a = a, b = b), predictvars = (y = KeepEach(),), iterations = 10)
        # We should get as many predictions as number of iterations (keep each )
        @test length(result_keep_each.predictions[:y]) === 10
        @test all(result_keep_each.predictions[:y] .== expected_prediction)
        # The posteriors however are not explicitly specified, should be derived from the number of iterations
        @test length(result_keep_each.posteriors[:θ]) === 10
        # Sine the data is missing the posteriors should be simply equal to the prior
        @test all(result_keep_each.posteriors[:θ] .== Beta(a, b))
    end
end