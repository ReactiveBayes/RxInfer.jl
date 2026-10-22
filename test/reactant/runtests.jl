# Compiling `infer` with Reactant: a linear Gaussian state-space model, unrolled over a few steps and
# compiled whole. Run with `make test-reactant`; it is not part of the test suite, since Reactant is a
# heavy dependency that changes often.

using Test, LinearAlgebra
using RxInfer, ReactiveMP, Reactant
import BayesBase

# Reactant's traced numbers are not `Real`, so the distributions here put no bound on their number type.

# A multivariate normal by its mean `μ` and covariance `Σ`.
struct TracedMvNormal{M <: AbstractVector, V <: AbstractMatrix}
    μ::M
    Σ::V
end

# A multivariate normal by its weighted mean `ξ = Λμ` and precision `Λ`.
struct TracedMvNormalWeighted{X <: AbstractVector, W <: AbstractMatrix}
    ξ::X
    Λ::W
end

BayesBase.mean(d::TracedMvNormal) = d.μ
BayesBase.cov(d::TracedMvNormal) = d.Σ

BayesBase.default_prod_rule(
    ::Type{<:TracedMvNormal}, ::Type{<:TracedMvNormalWeighted}
) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(
    ::Type{<:TracedMvNormalWeighted}, ::Type{<:TracedMvNormal}
) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(
    ::Type{<:TracedMvNormalWeighted}, ::Type{<:TracedMvNormalWeighted}
) = BayesBase.ClosedProd()

function BayesBase.prod(
        ::BayesBase.ClosedProd,
        prior::TracedMvNormal,
        likelihood::TracedMvNormalWeighted,
    )
    F = lu(I + prior.Σ * likelihood.Λ)
    return TracedMvNormal(F \ (prior.μ + prior.Σ * likelihood.ξ), F \ prior.Σ)
end
BayesBase.prod(
    strategy::BayesBase.ClosedProd,
    likelihood::TracedMvNormalWeighted,
    prior::TracedMvNormal,
) = BayesBase.prod(strategy, prior, likelihood)
BayesBase.prod(
    ::BayesBase.ClosedProd,
    left::TracedMvNormalWeighted,
    right::TracedMvNormalWeighted,
) = TracedMvNormalWeighted(left.ξ + right.ξ, left.Λ + right.Λ)

# `out ~ TracedMvNormal(μ, Σ)`, the prior.
@define_factor_node(
    node = TracedMvNormal, type = Stochastic, interfaces = [:out, :μ, :Σ]
)

@define_message_update_rule(
    node = TracedMvNormal,
    target = :out,
    args = (q[:μ]::PointMass, q[:Σ]::PointMass),
    body = (args) -> TracedMvNormal(mean(args.q[:μ]), mean(args.q[:Σ])),
)

# `out ~ N(A in, Σ)`: a transition, or an observation of `in`.
struct LinearGaussian end

@define_factor_node(
    node = LinearGaussian, type = Stochastic, interfaces = [:out, :in, :A, :Σ]
)

@define_message_update_rule(
    node = LinearGaussian,
    target = :out,
    args = (m[:in]::TracedMvNormal, q[:A]::PointMass, q[:Σ]::PointMass),
    body =
        (args) -> begin
        A, belief = mean(args.q[:A]), args.m[:in]
        TracedMvNormal(A * belief.μ, A * belief.Σ * A' + mean(args.q[:Σ]))
    end,
)

@define_message_update_rule(
    node = LinearGaussian,
    target = :in,
    args = (q[:out]::PointMass, q[:A]::PointMass, q[:Σ]::PointMass),
    body =
        (args) -> begin
        A, observation, noise = mean(args.q[:A]),
            mean(args.q[:out]),
            lu(mean(args.q[:Σ]))
        TracedMvNormalWeighted(
            A' * (noise \ observation), A' * (noise \ A)
        )
    end,
)

@define_message_update_rule(
    node = LinearGaussian,
    target = :in,
    args = (m[:out]::TracedMvNormalWeighted, q[:A]::PointMass, q[:Σ]::PointMass),
    body =
        (args) -> begin
        A, Σ, message = mean(args.q[:A]), mean(args.q[:Σ]), args.m[:out]
        F = lu(I + message.Λ * Σ)
        TracedMvNormalWeighted(
            A' * (F \ message.ξ), A' * (F \ message.Λ) * A
        )
    end,
)

@model function linear_gaussian_ssm(y, m0, V0, A, B, P, Q)
    x_prev ~ TracedMvNormal(m0, V0)
    for i in eachindex(y)
        x[i] ~ LinearGaussian(x_prev, A, P)
        y[i] ~ LinearGaussian(x[i], B, Q)
        x_prev = x[i]
    end
end

function smoothed_means(y, m0, V0, A, B, P, Q)
    result = infer(;
        model = linear_gaussian_ssm(;
            m0 = m0, V0 = V0, A = A, B = B, P = P, Q = Q
        ),
        data = (y = y,),
        returnvars = (x = KeepLast(),),
    )
    return map(mean, result.posteriors[:x])
end

θ = π / 35
A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
B, P, Q = Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2), Matrix(25.0I, 2, 2)
m0, V0 = zeros(2), Matrix(100.0I, 2, 2)
ys = [[cos(t / 10), sin(t / 10)] .* 10 for t in 1:5]

@testset "RxInfer compiled with Reactant" begin
    @testset "the extensions are loaded" begin
        @test Base.get_extension(RxInfer, :ReactantExt) !== nothing
        @test Base.get_extension(ReactiveMP, :ReactiveMPReactantExt) !== nothing
    end

    @testset "an unrolled model compiles and gives the same posteriors" begin
        # Compiled first: ReactiveMP's `factornode` caches how it built a node of each shape and builds
        # the next one from that plan, so a run before the compilation would spare the trace the code
        # that builds a node from scratch.
        traced = (
            map(Reactant.to_rarray, ys),
            Reactant.to_rarray.((m0, V0, A, B, P, Q))...,
        )
        compiled = @compile smoothed_means(traced...)
        computed = compiled(traced...)
        expected = smoothed_means(ys, m0, V0, A, B, P, Q)
        @test length(computed) == length(expected)
        @test all(isapprox.(Array.(computed), expected; atol = 1.0e-10))
    end
end

BayesBase.default_prod_rule(::Type{<:Gaussian}, ::Type{<:Gaussian}) = ClosedProd()
BayesBase.prod(::ClosedProd, a::Gaussian, b::Gaussian) =
    Gaussian((b.v .* a.μ .+ a.v .* b.μ) ./ (a.v .+ b.v), a.v .* b.v ./ (a.v .+ b.v))

@define_message_update_rule(
    node = +, target = (:in, k), args = (m[:out]::PointMass, m[:in][!k]::Gaussian),
    body = args -> begin
        other = args.m[:in][3 - k]
        Gaussian(mean(args.m[:out]) .- other.μ, other.v)
    end,
)

@model function sum_model(y, prior_a, prior_b)
    a ~ prior_a
    b ~ prior_b
    y ~ a + b
end

function posterior(y, ma, va, mb, vb)
    result = infer(
        model = sum_model(prior_a = Gaussian(ma, va), prior_b = Gaussian(mb, vb)),
        data = (; y), returnvars = (a = KeepLast(),), session = nothing
    )
    q = result.posteriors[:a]
    return q.μ, q.v
end

@testset "RxInfer compiled with Reactant for a node with variable number of interfaces" begin
    Reactant.set_default_backend("cpu")
    y, ma, va, mb, vb = Reactant.to_rarray.(([2.0], [0.0], [1.0], [0.0], [3.0]))
    compiled = Reactant.@compile sync = true posterior(y, ma, va, mb, vb)
    @assert Array.(compiled(y, ma, va, mb, vb)) == ([0.5], [0.75])
end
