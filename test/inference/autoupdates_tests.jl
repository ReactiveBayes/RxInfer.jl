@testitem "Autoupdates on a simple beta-bernoulli model" begin
    import RxInfer:
        numautoupdates, getautoupdate, getvarlabels, getarguments, AutoUpdateVariableLabel, AutoUpdateMapping, IndividualAutoUpdateSpecification, AutoUpdateFetchMarginalArgument

    @model function beta_bernoulli(y, a, b)
        θ ~ Beta(a, b)
        y ~ Bernoulli(θ)
    end

    @testset "Multiple variables on LHS, single variable on RHS" begin
        autoupdates = @autoupdates begin
            a, b = params(q(θ))
        end

        @test numautoupdates(autoupdates) == 1

        autoupdate1 = getautoupdate(autoupdates, 1)
        @test getvarlabels(autoupdate1) === (AutoUpdateVariableLabel(:a), AutoUpdateVariableLabel(:b))
        @test getmapping(autoupdate1) === AutoUpdateMapping(params, (AutoUpdateFetchMarginalArgument(:θ),))
    end

    @testset "`@autoupdates` should error if labels do not exist in the model specification" begin
        @test_broken false
    end
end

@testitem "q(x) and μ(x) are reserved functions" begin
    @test_throws "q(x) and μ(x) are reserved functions" eval(:(@autoupdates begin
        q(x) = 1
    end))

    @test_throws "q(x) and μ(x) are reserved functions" eval(:(@autoupdates begin
        μ(x) = 1
    end))
end

@testitem "autoupdate_argument_inexpr" begin
    import RxInfer: autoupdate_argument_inexpr

    @test autoupdate_argument_inexpr(:(q(x)))
    @test autoupdate_argument_inexpr(:(μ(x)))
    @test autoupdate_argument_inexpr(:(q(x[1])))
    @test autoupdate_argument_inexpr(:(μ(x[1])))
    @test autoupdate_argument_inexpr(:(1 + params(q(x))))
    @test autoupdate_argument_inexpr(:(1 + params(μ(x))))
    @test autoupdate_argument_inexpr(:(1 + params(q([1]))))
    @test autoupdate_argument_inexpr(:(1 + params(μ(x[1]))))
    @test autoupdate_argument_inexpr(:(f(q(x))))
    @test autoupdate_argument_inexpr(:(f(μ(x))))
    @test autoupdate_argument_inexpr(:(f(q(x[1]))))
    @test autoupdate_argument_inexpr(:(f(μ(x[2]))))
    @test !autoupdate_argument_inexpr(:(f(x)))
    @test !autoupdate_argument_inexpr(:(1 + 1))
    @test !autoupdate_argument_inexpr(:(y = x + 1))
    @test !autoupdate_argument_inexpr(:(f(x) = x + 1))
end

@testitem "is_autoupdate_mapping_expr" begin
    import RxInfer: is_autoupdate_mapping_expr

    @test is_autoupdate_mapping_expr(:(params(q(x))))
    @test is_autoupdate_mapping_expr(:(params(μ(x))))
    @test is_autoupdate_mapping_expr(:(f(q(x), q(x))))
    @test is_autoupdate_mapping_expr(:(f(q(x), μ(x))))
    @test is_autoupdate_mapping_expr(:(f(μ(x), q(x))))
    @test is_autoupdate_mapping_expr(:(f(μ(x), μ(x))))
end

@testitem "autoupdate_convert_mapping_expr" begin
    import RxInfer: autoupdate_convert_mapping_expr, AutoUpdateMapping, AutoUpdateFetchMarginalArgument, AutoUpdateFetchMessageArgument

    @test autoupdate_convert_mapping_expr(:(f(q(x)))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x),)))
    @test autoupdate_convert_mapping_expr(:(f(q(x[1])))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x, (1,)),)))
    @test autoupdate_convert_mapping_expr(:(f(q(x[1, 1])))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x, (1, 1)),)))
    @test autoupdate_convert_mapping_expr(:(g(μ(x)))) == :(RxInfer.AutoUpdateMapping(g, (RxInfer.AutoUpdateFetchMessageArgument(:x),)))
    @test autoupdate_convert_mapping_expr(:(g(μ(x[1])))) == :(RxInfer.AutoUpdateMapping(g, (RxInfer.AutoUpdateFetchMessageArgument(:x, (1,)),)))
    @test autoupdate_convert_mapping_expr(:(g(μ(x[1, 1])))) == :(RxInfer.AutoUpdateMapping(g, (RxInfer.AutoUpdateFetchMessageArgument(:x, (1, 1)),)))
    @test autoupdate_convert_mapping_expr(:(f(q(x), q(y)))) ==
        :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x), RxInfer.AutoUpdateFetchMarginalArgument(:y))))
    @test autoupdate_convert_mapping_expr(:(f(q(x), q(y)))) ==
        :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x), RxInfer.AutoUpdateFetchMarginalArgument(:y))))
    @test autoupdate_convert_mapping_expr(:(f(q(x), 2))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x), 2)))
    @test autoupdate_convert_mapping_expr(:(f(q(x), g(3, 2)))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x), g(3, 2))))
    # hierarchial call
    @test autoupdate_convert_mapping_expr(:(f(f(q(x))))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x),)),)))
    @test autoupdate_convert_mapping_expr(:(f(f(q(x), 1)))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x), 1)),)))
    @test autoupdate_convert_mapping_expr(:(f(f(q(x)), 1))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x),)), 1)))
end

@testitem "autoupdate_convert_labels_expr" begin
    import RxInfer: autoupdate_convert_labels_expr

    @test autoupdate_convert_labels_expr(:a) == :(RxInfer.AutoUpdateVariableLabel(:a))
    @test autoupdate_convert_labels_expr(:(a[1])) == :(RxInfer.AutoUpdateVariableLabel(:a, (1,)))
    @test autoupdate_convert_labels_expr(:(a[1, 1])) == :(RxInfer.AutoUpdateVariableLabel(:a, (1, 1)))
    @test autoupdate_convert_labels_expr(:(a, b)) == :((RxInfer.AutoUpdateVariableLabel(:a), RxInfer.AutoUpdateVariableLabel(:b)))
    @test autoupdate_convert_labels_expr(:(a, b[1])) == :((RxInfer.AutoUpdateVariableLabel(:a), RxInfer.AutoUpdateVariableLabel(:b, (1,))))
    @test autoupdate_convert_labels_expr(:(a, b[1, 1])) == :((RxInfer.AutoUpdateVariableLabel(:a), RxInfer.AutoUpdateVariableLabel(:b, (1, 1))))

    @test_throws "Cannot create variable label from expression `a + b`" autoupdate_convert_labels_expr(:(a + b))
    @test_throws "Cannot create variable label from expression `a * b`" autoupdate_convert_labels_expr(:(a * b))
    @test_throws "Cannot create variable label from expression `a = b`" autoupdate_convert_labels_expr(:(a = b))
    @test_throws "Cannot create variable label from expression `f(a)`" autoupdate_convert_labels_expr(:(f(a)))
end

@testitem "autoupdate_parse_autoupdate_specification_expr" begin
    import RxInfer: autoupdate_parse_autoupdate_specification_expr

    @test autoupdate_parse_autoupdate_specification_expr(:spec, :((a, b) = params(q(x)))) == :(
        spec = RxInfer.addspecification(
            spec, (RxInfer.AutoUpdateVariableLabel(:a), RxInfer.AutoUpdateVariableLabel(:b)), RxInfer.AutoUpdateMapping(params, (RxInfer.AutoUpdateFetchMarginalArgument(:x),))
        )
    )
end