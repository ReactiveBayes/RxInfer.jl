
@testitem "MultiAgentTrajectoryPlanning model should terminate and give results" begin
    # https://github.com/biaslab/MultiAgentTrajectoryPlanning/issues/4
    using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    # half space specification
    struct Halfspace end
    @node Halfspace Stochastic [out, a, σ2, γ]

    # rule specification
    @rule Halfspace(:out, Marginalisation) (q_a::PointMass, q_σ2::PointMass, q_γ::PointMass) = begin
        return NormalMeanVariance(mean(q_a) + mean(q_γ) * mean(q_σ2), mean(q_σ2))
    end

    struct ForcePointMass{V}
        v::V
    end

    @rule Halfspace(:σ2, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_a::PointMass, q_γ::PointMass) = begin
        return ForcePointMass(1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out)))
    end

    BayesBase.prod(::GenericProd, p::ForcePointMass, any) = PointMass(p.v)
    BayesBase.prod(::GenericProd, any, p::ForcePointMass) = PointMass(p.v)

    ReactiveMP.to_marginal(p::ForcePointMass) = PointMass(p.v)

    function h(y1, y2)
        r1 = 15
        r2 = 15
        return norm(y1 - y2) - r1 - r2
    end

    @model function switching_model(nr_steps, γ, ΔT, goals)

        # transition model
        A = [1 ΔT 0 0; 0 1 0 0; 0 0 1 ΔT; 0 0 0 1]
        B = [0 0; ΔT 0; 0 0; 0 ΔT]
        C = [1 0 0 0; 0 0 1 0]

        local y

        # single agent models
        for k in 1:2

            # prior on state
            x[k, 1] ~ MvNormalMeanCovariance(zeros(4), 1e2I)

            for t in 1:nr_steps

                # prior on controls
                u[k, t] ~ MvNormalMeanCovariance(zeros(2), 1e-2I)

                # state transition
                x[k, t + 1] ~ A * x[k, t] + B * u[k, t]

                # observation model
                y[k, t] ~ C * x[k, t + 1]
            end

            # goal priors (indexing reverse due to definition)
            goals[1, k] ~ MvNormalMeanCovariance(x[k, 1], 1e-5I)
            goals[2, k] ~ MvNormalMeanCovariance(x[k, nr_steps + 1], 1e-5I)
        end

        # multi-agent models
        for t in 1:nr_steps

            # observation constraint
            σ2[t] ~ GammaShapeRate(3 / 2, γ^2 / 2)
            d[t] ~ h(y[1, t], y[2, t])
            d[t] ~ Halfspace(0, σ2[t], γ)
        end
    end

    @constraints function switching_constraints()
        q(d, σ2) = q(d)q(σ2)
    end

    @meta function switching_meta()
        h() -> Linearization()
    end

    goals = hcat([
        # agent 1: start at (0,0) with 0 velocity, end at (0, 50) with 0 velocity
        [[0, 0, 0, 0], [0, 0, 50, 0]],
        # agent 2: start at (0,50) with 0 velocity, end at (0, 0) with 0 velocity
        [[0, 0, 50, 0], [0, 0, 0, 0]]
    ]...)

    @initialization function switching_initialization_1()
        q(σ2) = PointMass(1)
        μ(x) = MvNormalMeanCovariance(randn(4), 100I)
    end

    @initialization function switching_initialization_2()
        q(σ2) = PointMass(1)
        μ(y) = MvNormalMeanCovariance(randn(2), 100I)
    end

    for nr_steps in (50, 100), init in [switching_initialization_1, switching_initialization_2]
        result = infer(
            model          = switching_model(nr_steps = nr_steps, γ = 1, ΔT = 1),
            data           = (goals = goals,),
            constraints    = switching_constraints(),
            meta           = switching_meta(),
            initialization = init(),
            iterations     = 100,
            returnvars     = KeepLast(),
            showprogress   = false
        )
        @test mean(result.posteriors[:x][1, 1]) ≈ [0, 0, 0, 0] atol = 5e-1
        @test mean(result.posteriors[:x][1, nr_steps]) ≈ [0, 0, 50, 0] atol = 5e-1
        @test mean(result.posteriors[:x][2, 1]) ≈ [0, 0, 50, 0] atol = 5e-1
        @test mean(result.posteriors[:x][2, nr_steps]) ≈ [0, 0, 0, 0] atol = 5e-1
    end
end
