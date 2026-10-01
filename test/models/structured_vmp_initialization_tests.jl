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
    run(initialization) = infer(
        model = demo(),
        data = (ay = 1.0,),
        constraints = constraints,
        initialization = initialization,
        iterations = 50,
        free_energy = true,
    )

    reference, result = run(marginals), run(with_message)
    for name in (:x, :y, :z, :k, :w)
        @test mean(last(result.posteriors[name])) ≈
            mean(last(reference.posteriors[name])) atol = 1e-8
        @test var(last(result.posteriors[name])) ≈
            var(last(reference.posteriors[name])) atol = 1e-8
    end
    @test last(result.free_energy) ≈ last(reference.free_energy) atol = 1e-8
end

@testitem "a horizon with two steps of history and every marginal initialised" begin
    using MessagePassingRulesBase

    # y[t] ~ Hist(y[t-1], y[t-2], u[t], θ), a toy node with the shape of the Autoregressive Active
    # Inference example's MARX node: θ shared by every node, data only at the start, a goal at the
    # end, every marginal initialised, mean-field. Values computed from data reach the far end only
    # after several rounds, and each must restart the rules whose inputs are still initial: when a
    # rule restarted at most once, y and u were never updated.
    struct Hist end
    @define_factor_node(
        node = Hist,
        type = Stochastic,
        interfaces = [:out, :prev1, :prev2, :in, :θ]
    )
    m(args, k) = mean(args.q[k])
    @define_message_update_rule(
        node = Hist,
        target = :out,
        args = (q[:prev1]::Any, q[:prev2]::Any, q[:in]::Any, q[:θ]::Any),
        body =
            (args) -> NormalMeanVariance(
                m(args, :θ) * (m(args, :prev1) + m(args, :prev2)) / 2 +
                m(args, :in),
                1.0,
            ),
    )
    @define_message_update_rule(
        node = Hist,
        target = :prev1,
        args = (q[:out]::Any, q[:prev2]::Any, q[:in]::Any, q[:θ]::Any),
        body =
            (args) -> NormalMeanVariance(
                2 * (m(args, :out) - m(args, :in)) / m(args, :θ) -
                m(args, :prev2),
                4.0,
            ),
    )
    @define_message_update_rule(
        node = Hist,
        target = :prev2,
        args = (q[:out]::Any, q[:prev1]::Any, q[:in]::Any, q[:θ]::Any),
        body =
            (args) -> NormalMeanVariance(
                2 * (m(args, :out) - m(args, :in)) / m(args, :θ) -
                m(args, :prev1),
                4.0,
            ),
    )
    @define_message_update_rule(
        node = Hist,
        target = :in,
        args = (q[:out]::Any, q[:prev1]::Any, q[:prev2]::Any, q[:θ]::Any),
        body =
            (args) -> NormalMeanVariance(
                m(args, :out) -
                m(args, :θ) * (m(args, :prev1) + m(args, :prev2)) / 2,
                1.0,
            ),
    )
    @define_message_update_rule(
        node = Hist,
        target = :θ,
        args = (q[:out]::Any, q[:prev1]::Any, q[:prev2]::Any, q[:in]::Any),
        body = (args) -> NormalMeanVariance(0.5, 10.0),
    )

    @model function horizon(y1, y2, T)
        θ ~ NormalMeanVariance(1.0, 1.0)
        u[1] ~ NormalMeanVariance(0.0, 1.0)
        u[2] ~ NormalMeanVariance(0.0, 1.0)
        y[1] ~ Hist(y1, y2, u[1], θ)
        y[2] ~ Hist(y[1], y1, u[2], θ)
        for t in 3:T
            u[t] ~ NormalMeanVariance(0.0, 1.0)
            y[t] ~ Hist(y[t - 1], y[t - 2], u[t], θ)
        end
        y[T] ~ NormalMeanVariance(2.0, 0.1)
    end
    init = @initialization begin
        q(θ) = NormalMeanVariance(1.0, 1.0)
        q(y) = NormalMeanVariance(0.0, 10.0)
        q(u) = NormalMeanVariance(0.0, 10.0)
    end
    for T in (3, 5)
        result = infer(
            model = horizon(T = T),
            data = (y1 = 1.0, y2 = 0.5),
            constraints = MeanField(),
            initialization = init,
            iterations = 10,
        )
        @test all(q -> isfinite(mean(q)), last(result.posteriors[:y]))
        @test all(q -> isfinite(mean(q)), last(result.posteriors[:u]))
    end
end
