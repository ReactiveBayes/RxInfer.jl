@testitem "structured VMP with four clusters and an initial message (issue #344)" begin
    using GCVMessagePassingRules

    # Under q(x, y) q(k) q(z) q(w) the clusters wait on each other, so an initial message on `y`
    # once stopped inference: no posterior of w, k, z or x was ever computed. With it, inference
    # must reach the posteriors it reaches without it.
    @model function demo(ay)
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
    marginals = @initialization begin
        q(k) = NormalMeanVariance(0.0, 1.0)
        q(w) = NormalMeanVariance(0.0, 1.0)
        q(z) = NormalMeanVariance(0.0, 1.0)
    end
    with_message = @initialization begin
        q(k) = NormalMeanVariance(0.0, 1.0)
        q(w) = NormalMeanVariance(0.0, 1.0)
        q(z) = NormalMeanVariance(0.0, 1.0)
        μ(y) = NormalMeanVariance(0.0, 1.0)
    end
    run(initialization) = infer(model = demo(), data = (ay = 1.0,), constraints = constraints, initialization = initialization, iterations = 50, free_energy = true)

    reference, result = run(marginals), run(with_message)
    for name in (:x, :y, :z, :k, :w)
        @test mean(last(result.posteriors[name])) ≈ mean(last(reference.posteriors[name])) atol = 1e-8
        @test var(last(result.posteriors[name])) ≈ var(last(reference.posteriors[name])) atol = 1e-8
    end
    @test last(result.free_energy) ≈ last(reference.free_energy) atol = 1e-8
end
