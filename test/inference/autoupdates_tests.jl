@testitem "Autoupdates specs" begin
    import RxInfer:
        numautoupdates,
        getautoupdate,
        getvarlabels,
        AutoUpdateVariableLabel,
        AutoUpdateMapping,
        AutoUpdateSpecification,
        IndividualAutoUpdateSpecification,
        AutoUpdateFetchMarginalArgument,
        AutoUpdateFetchMessageArgument

    @testset "Single variables on LHS, single variable on RHS" begin
        f(qθ) = params(qθ)[1]
        g(qθ) = params(qθ)[2]

        autoupdates = @autoupdates begin
            a = f(q(θ))
            b = g(μ(θ))
        end

        @test numautoupdates(autoupdates) == 2

        autoupdate1 = getautoupdate(autoupdates, 1)
        @test getvarlabels(autoupdate1) === AutoUpdateVariableLabel(:a)
        @test getmapping(autoupdate1) === AutoUpdateMapping(f, (AutoUpdateFetchMarginalArgument(:θ),))

        autoupdate2 = getautoupdate(autoupdates, 2)
        @test getvarlabels(autoupdate2) === AutoUpdateVariableLabel(:b)
        @test getmapping(autoupdate2) === AutoUpdateMapping(g, (AutoUpdateFetchMessageArgument(:θ),))
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

    @testset "Single variables on LHS, complex expression on RHS" begin
        autoupdates = @autoupdates begin
            a = getindex(params(q(θ)), 1)
            b = getindex(params(q(θ)), 2)
        end

        @test numautoupdates(autoupdates) == 2

        autoupdate1 = getautoupdate(autoupdates, 1)
        @test getvarlabels(autoupdate1) === AutoUpdateVariableLabel(:a)
        @test getmapping(autoupdate1) === AutoUpdateMapping(getindex, (AutoUpdateMapping(params, (AutoUpdateFetchMarginalArgument(:θ),)), 1))

        autoupdate2 = getautoupdate(autoupdates, 2)
        @test getvarlabels(autoupdate2) === AutoUpdateVariableLabel(:b)
        @test getmapping(autoupdate2) === AutoUpdateMapping(getindex, (AutoUpdateMapping(params, (AutoUpdateFetchMarginalArgument(:θ),)), 2))
    end

    @testset "Complex expression inside `@autoupdates` function #1" begin
        for i in 1:3, j in 1:3
            # This is essentially equivalent to the following code:
            # autoupdates = @autoupdates begin
            #     x = mean(q(θ)) + (i * 2 + 3 * j)
            # end
            autoupdates = @autoupdates begin
                if false
                    error(1)
                end
                r = 0 # r = 2
                for k in 1:2
                    r = r + 1
                end
                d = 0 # d = 3
                while d < 3
                    d = d + 1
                end
                l(x, θ) = x * r + d * θ
                function returnzero()
                    return 0
                end
                c = l(i, j) - length("")
                x = mean(q(θ)) + (c + l(i, j) - l(i, j) + returnzero())
            end
            @test numautoupdates(autoupdates) == 1
            autoupdate1 = getautoupdate(autoupdates, 1)
            @test getvarlabels(autoupdate1) === AutoUpdateVariableLabel(:x)
            @test getmapping(autoupdate1) === AutoUpdateMapping(+, (AutoUpdateMapping(mean, (AutoUpdateFetchMarginalArgument(:θ),)), i * 2 + 3 * j))
        end
    end

    @testset "Complex expressions inside `@autoupdates` function #2" begin
        autoupdates = @autoupdates begin
            x = clamp(mean(q(z)), 0, 1)
        end
        @test numautoupdates(autoupdates) == 1
        autoupdate1 = getautoupdate(autoupdates, 1)
        @test getvarlabels(autoupdate1) === AutoUpdateVariableLabel(:x)
        @test getmapping(autoupdate1) === AutoUpdateMapping(clamp, (AutoUpdateMapping(mean, (AutoUpdateFetchMarginalArgument(:z),)), 0, 1))
    end

    @testset "Representation of `@autoupdates` should be easily readable" begin
        f(a, b) = mean(a) + mean(b)
        autoupdates = @autoupdates begin
            x = clamp(mean(q(z)), 0, 1 + 1)
            y[3] = f(q(g[1, 2]), μ(r[2])) + 3
        end
        @test repr(autoupdates) == """
        @autoupdates begin
            x = clamp(mean(q(z)), 0, 2)
            y[3] = +(f(q(g[1, 2]), μ(r[2])), 3)
        end
        """
    end

    @testset "`@autoupdates` can accept a function" begin 
        @autoupdates function myautoupdates(argument::Bool) 
            if argument
                x = mean(q(x))
            else
                y = mean(q(y))
            end
        end
        # To check the dispatch and complex function definition
        @autoupdates function myautoupdates(unused, input::S; keyword = input) where { S <: String }
            z = mean(q(z)) + keyword
        end

        autoupdates1_true = myautoupdates(true)

        @test numautoupdates(autoupdates1_true) == 1
        autoupdate1_true = getautoupdate(autoupdates1_true, 1)
        @test getvarlabels(autoupdate1_true) === AutoUpdateVariableLabel(:x)
        @test getmapping(autoupdate1_true) === AutoUpdateMapping(mean, (AutoUpdateFetchMarginalArgument(:x),))

        autoupdates1_false = myautoupdates(false)
        @test numautoupdates(autoupdates1_false) == 1
        autoupdate1_false = getautoupdate(autoupdates1_false, 1)
        @test getvarlabels(autoupdate1_false) === AutoUpdateVariableLabel(:y)
        @test getmapping(autoupdate1_false) === AutoUpdateMapping(mean, (AutoUpdateFetchMarginalArgument(:y),))

        autoupdates_string = myautoupdates(1, "hello")
        @test numautoupdates(autoupdates_string) == 1
        autoupdate1_string = getautoupdate(autoupdates_string, 1)
        @test getvarlabels(autoupdate1_string) === AutoUpdateVariableLabel(:z)
        @test getmapping(autoupdate1_string) === AutoUpdateMapping(+, (AutoUpdateMapping(mean, (AutoUpdateFetchMarginalArgument(:z),)), "hello"))
        
    end

    @testset "Check that the `@autoupdates` is type stable in simple cases" begin 
        function foo()
            @autoupdates begin
                x = clamp(mean(q(z)), 0, 1)
                y = clamp(mean(q(z)), 0, 1)
            end
        end
        @autoupdates function bar(input)
            x = clamp(mean(q(z)), 0, 1) + input
            y = clamp(mean(q(z)), 0, 1) - input
        end
        @test (@inferred(foo())) isa AutoUpdateSpecification
        @test (@inferred(bar(1))) isa AutoUpdateSpecification
    end
end

@testitem "Check that the `autoupdates` object is properly inferrable" begin
    import RxInfer: AutoUpdateSpecification, getvarlabels

    f1() = @autoupdates begin
        a = params(q(θ))
        b = params(μ(θ))
    end

    @test @inferred(f1()) isa AutoUpdateSpecification
    @test @inferred(getvarlabels(f1())) === (:a, :b)

    f2() = @autoupdates begin
        x = mean(q(θ)) - var(q(x))
        y = var(q(θ)) - mean(q(y))
    end

    @test @inferred(f2()) isa AutoUpdateSpecification
    @test @inferred(getvarlabels(f2())) === (:x, :y)

    f3() = @autoupdates begin
        a, b = mean_var(q(c))
        x = mean(q(z))
        y = var(q(z))
    end

    @test @inferred(f3()) isa AutoUpdateSpecification
    @test @inferred(getvarlabels(f3())) === (:a, :b, :x, :y)

    f4() = @autoupdates begin
        f(a) = 1, 2, 3, 4, 5, 6
        a, b, c, d, e, f = f(q(g))
    end

    @test @inferred(f4()) isa AutoUpdateSpecification
    @test @inferred(getvarlabels(f4())) === (:a, :b, :c, :d, :e, :f)
end

@testitem "Empty autoupdates are not allowed" begin
    @test_throws "`@autoupdates` did not find any auto-updates specifications. Check the documentation for more information." eval(:(@autoupdates begin end))
end

@testitem "`@autoupdates` requires a block of code" begin
    @test_throws "Autoupdates requires a block of code `begin ... end` or a full function definition as an input" eval(:(@autoupdates 1 + 1))
    @test_throws "Autoupdates requires a block of code `begin ... end` or a full function definition as an input" eval(:(@autoupdates a = q(θ)))
    @test_throws "Autoupdates requires a block of code `begin ... end` or a full function definition as an input" eval(:(@autoupdates q(θ)))
    @test_throws "Autoupdates requires a block of code `begin ... end` or a full function definition as an input" eval(:(@autoupdates θ))
end

@testitem "q(x) and μ(x) are reserved functions" begin
    @test_throws "q(x) and μ(x) are reserved functions" eval(:(@autoupdates begin
        q(x) = 1
    end))

    @test_throws "q(x) and μ(x) are reserved functions" eval(:(@autoupdates begin
        μ(x) = 1
    end))
end

@testitem "The `autoupdates` macro should support different options" begin
    import RxInfer: is_autoupdates_warn, is_autoupdates_strict

    autoupdates = @autoupdates begin
        a, b = params(q(θ))
    end

    @test is_autoupdates_warn(autoupdates)
    @test !is_autoupdates_strict(autoupdates)

    autoupdates_nowarn = @autoupdates [warn = false] begin
        a, b = params(q(θ))
    end

    @test !is_autoupdates_warn(autoupdates_nowarn)
    @test !is_autoupdates_strict(autoupdates_nowarn)

    autoupdates_strict = @autoupdates [strict = true] begin
        a, b = params(q(θ))
    end

    @test is_autoupdates_warn(autoupdates_strict)
    @test is_autoupdates_strict(autoupdates_strict)

    autoupdates_strict_nowarn = @autoupdates [strict = true, warn = false] begin
        a, b = params(q(θ))
    end

    @test !is_autoupdates_warn(autoupdates_strict_nowarn)
    @test is_autoupdates_strict(autoupdates_strict_nowarn)

    @test_throws "Unknown option for `@autoupdates`: hello = false. Supported options are [ `warn`, `strict` ]" eval(:(@autoupdates [hello = false] begin
        a, b = params(q(θ))
    end))

    @test_throws "Invalid value for `warn` option. Expected `true` or `false`." eval(:(@autoupdates [warn = "hello"] begin
        a, b = params(q(θ))
    end))

    @test_throws "Invalid value for `strict` option. Expected `true` or `false`." eval(:(@autoupdates [strict = "hello"] begin
        a, b = params(q(θ))
    end))
end

@testitem "The `autoupdates` structure can be prepared for a specific model #1 - Beta Bernoulli" begin
    import RxInfer: prepare_for_model

    @model function beta_bernoulli(a, b, y)
        θ ~ Beta(a, b)
        y ~ Bernoulli(θ)
    end

    autoupdates = @autoupdates begin
        a, b = params(q(θ))
    end

    autoupdates_nowarn = @autoupdates [warn = false] begin
        a, b = params(q(θ))
    end

    autoupdates_strict = @autoupdates [strict = true] begin
        a, b = params(q(θ))
    end

    autoupdates_strict_nowarn = @autoupdates [strict = true, warn = false] begin
        a, b = params(q(θ))
    end

    @testset "Warning/error if the varlabel in `autoupdates` have been reserved with constants" begin
        model = beta_bernoulli(a = 1, b = 1)
        @test_logs(
            (:warn, r".*Autoupdates defines an update for `a`\, but `a` has been reserved in the model as a constant.*Use `warn = false` option to supress the warning.*Use `strict = true` option to turn the warning into an error.*"),
            (:warn, r".*Autoupdates defines an update for `b`\, but `b` has been reserved in the model as a constant.*Use `warn = false` option to supress the warning.*Use `strict = true` option to turn the warning into an error.*"),
            prepare_for_model(autoupdates, model)
        )
        @test_logs prepare_for_model(autoupdates_nowarn, model)
        @test_throws "Autoupdates defines an update for `a`, but `a` has been reserved in the model as a constant." prepare_for_model(autoupdates_strict, model)
        @test_throws "Autoupdates defines an update for `a`, but `a` has been reserved in the model as a constant." prepare_for_model(autoupdates_strict_nowarn, model)
    end
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