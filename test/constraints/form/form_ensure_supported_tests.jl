@testitem "Tests for `EnsureSupportedFunctionalForm" begin
    import RxInfer: EnsureSupportedFunctionalForm
    import ReactiveMP: default_form_check_strategy, default_prod_constraint, constrain_form
    import BayesBase: PointMass, ProductOf, LinearizedProductOf

    # In principle any object is supported except `ProductOf` and `LinearizedProductOf` from `BayesBase`
    # Those are supposed to be passed to the functional form constraint

    for prefix in (:q, :μ), name in (:a, :b)
        @test default_form_check_strategy(EnsureSupportedFunctionalForm(prefix, name)) === FormConstraintCheckLast()
        @test default_prod_constraint(EnsureSupportedFunctionalForm(prefix, name)) === GenericProd()

        @testset let constraint = EnsureSupportedFunctionalForm(prefix, name)
            @test constrain_form(constraint, PointMass(1)) === PointMass(1)
            @test_throws Exception constrain_form(constraint, ProductOf(PointMass(1), PointMass(2)))
            @test_throws Exception constrain_form(constraint, LinearizedProductOf([PointMass(1), PointMass(2)], 2))
        end
    end
end