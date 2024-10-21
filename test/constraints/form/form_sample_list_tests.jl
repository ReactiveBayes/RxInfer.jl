@testitem "PointMassFormConstraint" begin
    using DomainSets, StableRNGs, DomainSets, Distributions, Random, LinearAlgebra
    import RxInfer: SampleListFormConstraint, constrain_form

    @testset "approximation for generic univariate f" begin
        irng = StableRNG(42)

        for _ in 1:3, nsamples in (5000, 10000)
            constraint = SampleListFormConstraint(nsamples)

            d = NormalMeanVariance(randn(irng), rand(irng))
            f = ContinuousUnivariateLogPdf((x) -> 0)

            prod_df = prod(GenericProd(), d, f)
            prod_fd = prod(GenericProd(), f, d)

            for prod in (prod_df, prod_fd)
                q = constrain_form(constraint, prod)

                @test q isa SampleList
                @test length(q) === nsamples
                @test mean(q) ≈ mean(d) atol = 1e-1
                @test var(q) ≈ var(d) atol = 1e-1
            end
        end
    end

    @testset "approximation for generic multivariate f" begin
        irng = StableRNG(42)

        for s in 2:4, nsamples in (5000, 10000)
            constraint = SampleListFormConstraint(nsamples)

            d = MvNormalMeanCovariance(randn(irng, Float64, s), Diagonal(rand(irng, Float64, s)))
            f = ContinuousMultivariateLogPdf(s, (x) -> 0)

            prod_df = prod(GenericProd(), d, f)
            prod_fd = prod(GenericProd(), f, d)

            for prod in (prod_df, prod_fd)
                q = constrain_form(constraint, prod)

                @test q isa SampleList
                @test length(q) === nsamples
                @test mean(q) ≈ mean(d) atol = 1e-1
                @test var(q) ≈ var(d) atol = 1e-1
            end
        end
    end

    @testset "approximation when left or right is LinearizedProductOf" begin
        using LinearAlgebra
        left = MvNormalMeanCovariance(randn(2), Diagonal(rand(2)))

        right = LinearizedProductOf([ContinuousMultivariateLogPdf(2, (x) -> rand()) for _ in 1:10], 10)
        constraint = SampleListFormConstraint(100)
        q = constrain_form(constraint, prod(GenericProd(), left, right))
        @test q isa SampleList
    end
end
