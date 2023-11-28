@testitem "PointMassFormConstraint" begin
    using DomainSets, StableRNGs, DomainSets, Distributions, Random, LinearAlgebra
    import RxInfer: SampleListFormConstraint, is_point_mass_form_constraint, constrain_form
    @testset "is_point_mass_form_constraint" begin
        @test !is_point_mass_form_constraint(SampleListFormConstraint(100))
    end

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
end
