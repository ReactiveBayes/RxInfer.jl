@testitem "GCV with 3+ mean-field clusters should support an initialized message (issue #344)" begin
    # Regression test for https://github.com/ReactiveBayes/RxInfer.jl/issues/344
    # Structured VMP around a `GCV` node with the `q(x, y)q(k)q(z)q(w)` factorization
    # deadlocked when a message was initialized in addition to the required marginals:
    # the first outbound message computation consumed only provisional (`is_initial`)
    # marginals and the dependencies never became consumable again, failing inference
    # with a `Variables [ w, k, z, x ] have not been updated` error.
    # Requires ReactiveMP with the fix from
    # https://github.com/ReactiveBayes/ReactiveMP.jl/pull/620

    @model function gcv_single_step(ay)
        x ~ NormalMeanVariance(0.0, 1.0)
        z ~ NormalMeanVariance(0.0, 1.0)
        k ~ NormalMeanVariance(0.0, 1.0)
        w ~ NormalMeanVariance(0.0, 1.0)
        y ~ GCV(x, z, k, w)
        ay ~ NormalMeanVariance(y, 1.0)
    end

    constraints = @constraints begin
        q(x, y, z, k, w) = q(x, y)q(k)q(z)q(w)
    end

    init_marginals_only = @initialization begin
        q(k) = NormalMeanVariance(0.0, 1.0)
        q(w) = NormalMeanVariance(0.0, 1.0)
        q(z) = NormalMeanVariance(0.0, 1.0)
    end

    # The extra initialized message used to deadlock the inference procedure
    init_with_message = @initialization begin
        q(k) = NormalMeanVariance(0.0, 1.0)
        q(w) = NormalMeanVariance(0.0, 1.0)
        q(z) = NormalMeanVariance(0.0, 1.0)
        μ(y) = NormalMeanVariance(0.0, 1.0)
    end

    function gcv_inference(init)
        return infer(
            model = gcv_single_step(),
            data = (ay = 1.0,),
            constraints = constraints,
            initialization = init,
            iterations = 20,
            free_energy = true,
        )
    end

    results_marginals_only = gcv_inference(init_marginals_only)
    results_with_message = gcv_inference(init_with_message)

    for results in (results_marginals_only, results_with_message)
        for v in (:x, :z, :k, :w)
            @test length(results.posteriors[v]) === 20
        end
        @test mean(results.posteriors[:x][end]) ≈ 0.3972145516 atol = 1e-6
        @test var(results.posteriors[:x][end]) ≈ 0.6027854484 atol = 1e-6
        @test last(results.free_energy) ≈ 1.9679321014 atol = 1e-6
    end

    # Both initializations must converge to the same posteriors
    for v in (:x, :z, :k, :w)
        @test mean(results_marginals_only.posteriors[v][end]) ≈
            mean(results_with_message.posteriors[v][end]) atol = 1e-8
        @test var(results_marginals_only.posteriors[v][end]) ≈
            var(results_with_message.posteriors[v][end]) atol = 1e-8
    end
end
