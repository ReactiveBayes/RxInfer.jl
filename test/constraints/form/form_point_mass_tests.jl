@testitem "PointMassFormConstraint" begin
    using Test
    using RxInfer, LinearAlgebra
    using Random, StableRNGs, DomainSets, Distributions

    struct MyDistributionWithMode <: ContinuousUnivariateDistribution
        mode::Float64
    end

    # We are testing specifically that the point mass optimizer does not call `logpdf` and 
    # chooses a fast path with `mode` for `<: Distribution` objects
    Distributions.logpdf(::MyDistributionWithMode, _) = error("This should not be called")
    Distributions.mode(d::MyDistributionWithMode)     = d.mode
    Distributions.support(::MyDistributionWithMode)   = RealInterval(-Inf, Inf)

    const arbitrary_dist_1 = ContinuousUnivariateLogPdf(RealLine(), (x) -> logpdf(NormalMeanVariance(0, 1), x))
    const arbitrary_dist_2 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> logpdf(Gamma(1, 1), x))
    const arbitrary_dist_3 = ContinuousUnivariateLogPdf(RealLine(), (x) -> logpdf(NormalMeanVariance(-10, 10), x))
    const arbitrary_dist_4 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> logpdf(GammaShapeRate(100, 10), x))
    const arbitrary_dist_5 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> logpdf(GammaShapeRate(100, 100), x))

    @testset "is_point_mass_form_constraint" begin
        @test is_point_mass_form_constraint(PointMassFormConstraint())
    end

    @testset "boundaries" begin
        constraint = PointMassFormConstraint()

        @test call_boundaries(constraint, arbitrary_dist_1) === (-Inf, Inf)
        @test call_boundaries(constraint, arbitrary_dist_2) === (0.0, Inf)

        bm_constraint = PointMassFormConstraint(boundaries = (args...) -> (-1.0, 1.0))

        @test call_boundaries(bm_constraint, arbitrary_dist_1) === (-1.0, 1.0)
        @test call_boundaries(bm_constraint, arbitrary_dist_2) === (-1.0, 1.0)
    end

    @testset "starting point" begin
        constraint = PointMassFormConstraint()

        @test call_starting_point(constraint, arbitrary_dist_1) == [0.0]
        @test_throws ErrorException call_starting_point(constraint, arbitrary_dist_2)

        bm_constraint = PointMassFormConstraint(starting_point = (args...) -> [1.0])

        @test call_starting_point(bm_constraint, arbitrary_dist_1) == [1.0]
        @test call_starting_point(bm_constraint, arbitrary_dist_2) == [1.0]
    end

    @testset "optimizer" begin
        constraint = PointMassFormConstraint()

        @test isapprox(mean(constrain_form(constraint, arbitrary_dist_1)), 0.0, atol = 0.1)
        @test isapprox(mean(constrain_form(constraint, arbitrary_dist_3)), -10.0, atol = 0.1)
        @test_throws ErrorException constrain_form(constraint, arbitrary_dist_2)

        gopt_constraint = PointMassFormConstraint(starting_point = (args...) -> [1.0])

        @test isapprox(mean(constrain_form(gopt_constraint, arbitrary_dist_4)), 10, atol = 0.1)
        @test isapprox(mean(constrain_form(gopt_constraint, arbitrary_dist_5)), 1, atol = 0.1)

        bm_constraint = PointMassFormConstraint(optimizer = (args...) -> PointMass(10.0))

        @test constrain_form(bm_constraint, arbitrary_dist_1) == PointMass(10.0)
        @test constrain_form(bm_constraint, arbitrary_dist_2) == PointMass(10.0)
    end

    @testset "fast path for Distribution" begin
        constraint = PointMassFormConstraint()

        for mode in randn(4)
            @test mean(constrain_form(constraint, MyDistributionWithMode(mode))) === mode
        end
    end

    @testset "optimizer for generic f" begin
        irng = StableRNG(42)

        for _ in 1:3
            constraint = PointMassFormConstraint()

            d1 = NormalMeanVariance(10randn(irng), 10rand(irng))
            d2 = NormalMeanVariance(10randn(irng), 10rand(irng))

            f = ContinuousUnivariateLogPdf((x) -> logpdf(d1, x) + logpdf(d2, x))

            opt = constrain_form(constraint, f)

            analytical = prod(PreserveTypeProd(Distribution), d1, d2)

            @test isapprox(mode(opt), mode(analytical), atol = 1e-8)
        end

        for _ in 1:3
            constraint = PointMassFormConstraint(boundaries = (args...) -> (0.0, Inf), starting_point = (args...) -> [1.0])

            d1 = Gamma(10rand(irng) + 1, 10rand(irng) + 1)
            d2 = Gamma(10rand(irng) + 1, 10rand(irng) + 1)

            f = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> logpdf(d1, x) + logpdf(d2, x))

            opt = constrain_form(constraint, f)

            analytical = prod(PreserveTypeProd(Distribution), d1, d2)

            @test isapprox(mode(opt), mode(analytical), atol = 1e-4)
        end
    end
end
