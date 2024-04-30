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
            x = clamp.(mean.(q(z)), 0, 1 + 1)
            y[3] = f(q(g[1, 2]), μ(r[2])) + 3
        end
        @test repr(autoupdates) == """
        @autoupdates begin
            x = clamp.(mean.(q(z)), 0, 2)
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
        @autoupdates function myautoupdates(unused, input::S; keyword = input) where {S <: String}
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

    @autoupdates [warn = false] function autoupdates_nowarn_function()
        a, b = params(q(θ))
    end

    @test !is_autoupdates_warn(autoupdates_nowarn)
    @test !is_autoupdates_strict(autoupdates_nowarn)
    @test !is_autoupdates_warn(autoupdates_nowarn_function())
    @test !is_autoupdates_strict(autoupdates_nowarn_function())

    autoupdates_strict = @autoupdates [strict = true] begin
        a, b = params(q(θ))
    end

    @autoupdates [strict = true] function autoupdates_strict_function()
        a, b = params(q(θ))
    end

    @test is_autoupdates_warn(autoupdates_strict)
    @test is_autoupdates_strict(autoupdates_strict)
    @test is_autoupdates_warn(autoupdates_strict_function())
    @test is_autoupdates_strict(autoupdates_strict_function())

    autoupdates_strict_nowarn = @autoupdates [strict = true, warn = false] begin
        a, b = params(q(θ))
    end

    @autoupdates [strict = true, warn = false] function autoupdates_strict_nowarn_function()
        a, b = params(q(θ))
    end

    @test !is_autoupdates_warn(autoupdates_strict_nowarn)
    @test is_autoupdates_strict(autoupdates_strict_nowarn)
    @test !is_autoupdates_warn(autoupdates_strict_nowarn_function())
    @test is_autoupdates_strict(autoupdates_strict_nowarn_function())

    @test_throws "Unknown option for `@autoupdates`: hello = false. Supported options are [ `warn`, `strict` ]" eval(:(@autoupdates [hello = false] begin
        a, b = params(q(θ))
    end))
    @test_throws "Unknown option for `@autoupdates`: hello = false. Supported options are [ `warn`, `strict` ]" eval(:(@autoupdates [hello = false] function hello_function()
        a, b = params(q(θ))
    end))

    @test_throws "Invalid value for `warn` option. Expected `true` or `false`." eval(:(@autoupdates [warn = "hello"] begin
        a, b = params(q(θ))
    end))
    @test_throws "Invalid value for `warn` option. Expected `true` or `false`." eval(:(@autoupdates [warn = "hello"] function hello_function()
        a, b = params(q(θ))
    end))

    @test_throws "Invalid value for `strict` option. Expected `true` or `false`." eval(:(@autoupdates [strict = "hello"] begin
        a, b = params(q(θ))
    end))
    @test_throws "Invalid value for `strict` option. Expected `true` or `false`." eval(:(@autoupdates [strict = "hello"] function hello_function()
        a, b = params(q(θ))
    end))
end

@testitem "The `autoupdates` structure can be prepared for a specific model #1 - Beta Bernoulli" begin
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
        import RxInfer: check_model_generator_compatibility

        model = beta_bernoulli(a = 1, b = 1)
        @test_logs(
            (
                :warn,
                r".*Autoupdates defines an update for `a`\, but `a` has been reserved in the model as a constant.*Use `warn = false` option to supress the warning.*Use `strict = true` option to turn the warning into an error.*"
            ),
            (
                :warn,
                r".*Autoupdates defines an update for `b`\, but `b` has been reserved in the model as a constant.*Use `warn = false` option to supress the warning.*Use `strict = true` option to turn the warning into an error.*"
            ),
            check_model_generator_compatibility(autoupdates, model)
        )
        @test_logs check_model_generator_compatibility(autoupdates_nowarn, model)
        @test_throws "Autoupdates defines an update for `a`, but `a` has been reserved in the model as a constant." check_model_generator_compatibility(autoupdates_strict, model)
        @test_throws "Autoupdates defines an update for `a`, but `a` has been reserved in the model as a constant." check_model_generator_compatibility(
            autoupdates_strict_nowarn, model
        )
    end

    @testset "Create deferred data handlers from the `@autoupdates` specification" begin
        import RxInfer: DeferredDataHandler, autoupdates_data_handlers

        for autoupdate in (autoupdates, autoupdates_nowarn, autoupdates_strict, autoupdates_strict_nowarn)
            @test @inferred(autoupdates_data_handlers(autoupdate)) === (a = DeferredDataHandler(), b = DeferredDataHandler())
        end

        autoupdates_with_indices_1 = @autoupdates begin
            ins[1], ins[2] = collect(params(q(θ)))
        end

        autoupdates_with_indices_2 = @autoupdates begin
            ins[1], a, b, ins[2] = collect(params(q(θ)))
        end

        @test @inferred(autoupdates_data_handlers(autoupdates_with_indices_1)) === (ins = DeferredDataHandler(),)
        @test @inferred(autoupdates_data_handlers(autoupdates_with_indices_2)) === (ins = DeferredDataHandler(), a = DeferredDataHandler(), b = DeferredDataHandler())
    end

    @testset "Check that variables have been fetched correctly" begin
        import RxInfer:
            DeferredDataHandler,
            create_model,
            ReactiveMPInferencePlugin,
            ReactiveMPInferenceOptions,
            numautoupdates,
            getvarlabels,
            getmapping,
            getmappingfn,
            getarguments,
            getautoupdate,
            FetchRecentArgument,
            AutoUpdateMapping,
            prepare_autoupdates_for_model,
            getvariable,
            run_autoupdate!
        import GraphPPL: VariationalConstraintsPlugin, PluginsCollection, with_plugins, getextra

        # Here we simply want to add some complex `autoupdates`, which are all essentially equivalent
        # The purpose is just to verify certain possibilities, even though they don't have a lot of sense in real scenarious
        autoupdates_extra_1 = @autoupdates begin
            function f(qθ1, qθ2)
                @test params(qθ1) === params(qθ2)
                return params(qθ1)
            end
            a, b = f(q(θ), q(θ))
        end

        autoupdates_extra_2 = @autoupdates begin
            function f(qθ, arg)
                return params(qθ) .+ arg .- arg
            end
            a, b = f(q(θ), 1)
        end

        autoupdates_extra_3 = @autoupdates begin
            a = getindex(params(q(θ)), 1)
            b = getindex(params(q(θ)), 2)
        end

        @autoupdates function autoupdates_extra_4_fn(i, j)
            a = getindex(params(q(θ)), i)
            b = getindex(params(q(θ)), j)
        end

        autoupdates_extra_4 = autoupdates_extra_4_fn(1, 2)

        @autoupdates function autoupdates_extra_5_fn(f)
            a, b = f(q(θ))
        end

        autoupdates_extra_5 = autoupdates_extra_5_fn((q) -> params(q))

        autoupdates_to_test = (
            autoupdates,
            autoupdates_nowarn,
            autoupdates_strict,
            autoupdates_strict_nowarn,
            autoupdates_extra_1,
            autoupdates_extra_2,
            autoupdates_extra_3,
            autoupdates_extra_4,
            autoupdates_extra_5
        )

        for autoupdate in autoupdates_to_test
            extra_data_handlers = autoupdates_data_handlers(autoupdate)
            data_handlers = (y = DeferredDataHandler(), extra_data_handlers...)
            options = convert(ReactiveMPInferenceOptions, (;))
            plugins = PluginsCollection(VariationalConstraintsPlugin(), ReactiveMPInferencePlugin(options))
            model = create_model(with_plugins(beta_bernoulli(), plugins) | data_handlers)
            variable_a = getvariable(getindex(getvardict(model), :a))
            variable_b = getvariable(getindex(getvardict(model), :b))
            variable_y = getvariable(getindex(getvardict(model), :y))
            variable_θ = getvariable(getindex(getvardict(model), :θ))

            @test variable_a != variable_b

            autoupdates_for_model = prepare_autoupdates_for_model(autoupdate, model)

            marginals_θ = []
            updates_for_a = []
            updates_for_b = []
            updates_for_y = []
            subscription_marginal_θ = subscribe!(getmarginal(variable_θ, IncludeAll()), (qθ) -> push!(marginals_θ, qθ))
            subscription_updates_a = subscribe!(getmarginal(variable_a, IncludeAll()), (a) -> push!(updates_for_a, a))
            subscription_updates_b = subscribe!(getmarginal(variable_b, IncludeAll()), (b) -> push!(updates_for_b, b))
            subscription_updates_y = subscribe!(getmarginal(variable_y, IncludeAll()), (y) -> push!(updates_for_y, y))

            update!(variable_a, 1)
            update!(variable_b, 2)
            update!(variable_y, 1)

            @test length(marginals_θ) === 1
            @test length(updates_for_a) === 1
            @test length(updates_for_b) === 1
            @test length(updates_for_y) === 1
            @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0)]
            @test ReactiveMP.getdata.(updates_for_a) == [PointMass(1)]
            @test ReactiveMP.getdata.(updates_for_b) == [PointMass(2)]
            @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1)]

            run_autoupdate!(autoupdates_for_model)

            @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0)] # The marginal should not update at this point, because `y` has not been updated
            @test ReactiveMP.getdata.(updates_for_a) == [PointMass(1), PointMass(2.0)]
            @test ReactiveMP.getdata.(updates_for_b) == [PointMass(2), PointMass(2.0)]
            @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1)]

            update!(variable_y, 0)

            @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0), Beta(2.0, 3.0)]
            @test ReactiveMP.getdata.(updates_for_a) == [PointMass(1), PointMass(2.0)]
            @test ReactiveMP.getdata.(updates_for_b) == [PointMass(2), PointMass(2.0)]
            @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1), PointMass(0)]

            unsubscribe!([subscription_marginal_θ, subscription_updates_a, subscription_updates_b, subscription_updates_y])
        end
    end
end

@testitem "The `autoupdates` structure can be prepared for a specific model #2 - Beta Bernoulli" begin
    import RxInfer:
        DeferredDataHandler,
        create_model,
        ReactiveMPInferencePlugin,
        ReactiveMPInferenceOptions,
        numautoupdates,
        getvarlabels,
        getmapping,
        getmappingfn,
        getarguments,
        getautoupdate,
        FetchRecentArgument,
        AutoUpdateMapping,
        prepare_autoupdates_for_model,
        getvariable,
        autoupdates_data_handlers,
        run_autoupdate!
    import GraphPPL: VariationalConstraintsPlugin, PluginsCollection, with_plugins, getextra

    @model function beta_bernoulli_vector_based(a, b, y)
        θ[1] ~ Beta(a, b)
        y ~ Bernoulli(θ[1])
    end

    autoupdates_1 = @autoupdates begin
        a, b = params(getindex(q(θ), 1))
    end

    autoupdates_2 = @autoupdates begin
        a, b = params(q(θ[1]))
    end

    autoupdates_3 = @autoupdates begin
        foo(qθ) = params(qθ[1])
        a, b = foo(q(θ))
    end

    autoupdates_4 = @autoupdates begin
        a, b = getindex(params.(q(θ)), 1)
    end

    for autoupdate in (autoupdates_1, autoupdates_2, autoupdates_3, autoupdates_4)
        extra_data_handlers = autoupdates_data_handlers(autoupdate)
        data_handlers = (y = DeferredDataHandler(), extra_data_handlers...)
        options = convert(ReactiveMPInferenceOptions, (;))
        plugins = PluginsCollection(VariationalConstraintsPlugin(), ReactiveMPInferencePlugin(options))
        model = create_model(with_plugins(beta_bernoulli_vector_based(), plugins) | data_handlers)
        variable_a = getvariable(getindex(getvardict(model), :a))
        variable_b = getvariable(getindex(getvardict(model), :b))
        variable_y = getvariable(getindex(getvardict(model), :y))
        variable_θ = getvariable(getindex(getvardict(model), :θ))

        autoupdates_for_model = prepare_autoupdates_for_model(autoupdate, model)
        @test numautoupdates(autoupdates_for_model) == 1
        autoupdate1 = getautoupdate(autoupdates_for_model, 1)

        marginals_θ = []
        updates_for_a = []
        updates_for_b = []
        updates_for_y = []
        subscription_marginal_θ = subscribe!(getmarginal(variable_θ[1], IncludeAll()), (qθ) -> push!(marginals_θ, qθ))
        subscription_updates_a = subscribe!(getmarginal(variable_a, IncludeAll()), (a) -> push!(updates_for_a, a))
        subscription_updates_b = subscribe!(getmarginal(variable_b, IncludeAll()), (b) -> push!(updates_for_b, b))
        subscription_updates_y = subscribe!(getmarginal(variable_y, IncludeAll()), (y) -> push!(updates_for_y, y))

        update!(variable_a, 1)
        update!(variable_b, 2)
        update!(variable_y, 1)

        @test length(marginals_θ) === 1
        @test length(updates_for_a) === 1
        @test length(updates_for_b) === 1
        @test length(updates_for_y) === 1
        @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0)]
        @test ReactiveMP.getdata.(updates_for_a) == [PointMass(1)]
        @test ReactiveMP.getdata.(updates_for_b) == [PointMass(2)]
        @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1)]
        @test fetch(autoupdate1) == (2.0, 2.0)

        run_autoupdate!(autoupdates_for_model)

        @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0)] # The marginal should not update at this point, because `y` has not been updated
        @test ReactiveMP.getdata.(updates_for_a) == [PointMass(1), PointMass(2.0)]
        @test ReactiveMP.getdata.(updates_for_b) == [PointMass(2), PointMass(2.0)]
        @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1)]

        update!(variable_y, 0)

        @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0), Beta(2.0, 3.0)]
        @test ReactiveMP.getdata.(updates_for_a) == [PointMass(1), PointMass(2.0)]
        @test ReactiveMP.getdata.(updates_for_b) == [PointMass(2), PointMass(2.0)]
        @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1), PointMass(0)]

        unsubscribe!([subscription_marginal_θ, subscription_updates_a, subscription_updates_b, subscription_updates_y])
    end
end

@testitem "The `autoupdates` structure can be prepared for a specific model #3 - Beta Bernoulli" begin
    import RxInfer:
        DeferredDataHandler,
        create_model,
        ReactiveMPInferencePlugin,
        ReactiveMPInferenceOptions,
        numautoupdates,
        getvarlabels,
        getmapping,
        getmappingfn,
        getarguments,
        getautoupdate,
        FetchRecentArgument,
        AutoUpdateMapping,
        prepare_autoupdates_for_model,
        getvariable,
        autoupdates_data_handlers,
        run_autoupdate!
    import GraphPPL: VariationalConstraintsPlugin, PluginsCollection, with_plugins, getextra

    @model function beta_bernoulli_vector_based_args(ins, y)
        θ[1, 1] ~ Beta(ins[1], ins[2])
        y ~ Bernoulli(θ[1, 1])
    end

    autoupdates_1 = @autoupdates begin
        ins = collect(params(q(θ[1, 1])))
    end

    autoupdates_2 = @autoupdates begin
        ins[1] = getindex(params(q(θ[1, 1])), 1)
        ins[2] = getindex(params(q(θ[1, 1])), 2)
    end

    autoupdates_3 = @autoupdates begin
        ins[1], ins[2] = params(q(θ[1, 1]))
    end

    autoupdates_4 = @autoupdates begin
        ins[1], ins[2] = getindex(params.(q(θ)), 1, 1)
    end

    autoupdates_5 = @autoupdates begin
        ins = collect(getindex(params.(q(θ)), 1, 1))
    end

    autoupdates_6 = @autoupdates begin
        ins[1] = getindex(getindex(params.(q(θ)), 1, 1), 1)
        ins[2] = getindex(getindex(params.(q(θ)), 1, 1), 2)
    end

    for autoupdate in (autoupdates_1, autoupdates_2, autoupdates_3, autoupdates_4, autoupdates_5, autoupdates_6)
        extra_data_handlers = autoupdates_data_handlers(autoupdate)
        data_handlers = (y = DeferredDataHandler(), extra_data_handlers...)
        options = convert(ReactiveMPInferenceOptions, (;))
        plugins = PluginsCollection(VariationalConstraintsPlugin(), ReactiveMPInferencePlugin(options))
        model = create_model(with_plugins(beta_bernoulli_vector_based_args(), plugins) | data_handlers)
        variable_ins = getvariable(getindex(getvardict(model), :ins))
        variable_y = getvariable(getindex(getvardict(model), :y))
        variable_θ = getvariable(getindex(getvardict(model), :θ))

        autoupdates_for_model = prepare_autoupdates_for_model(autoupdate, model)

        marginals_θ = []
        updates_for_ins = []
        updates_for_y = []
        subscription_marginal_θ = subscribe!(getmarginal(variable_θ[1, 1], IncludeAll()), (qθ) -> push!(marginals_θ, qθ))
        subscription_updates_ins = subscribe!(getmarginals(variable_ins, IncludeAll()), (ins) -> push!(updates_for_ins, ins))
        subscription_updates_y = subscribe!(getmarginal(variable_y, IncludeAll()), (y) -> push!(updates_for_y, y))

        update!(variable_ins, [1, 2])
        update!(variable_y, 1)

        @test length(marginals_θ) === 1
        @test length(updates_for_ins) === 1
        @test length(updates_for_y) === 1
        @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0)]
        @test ReactiveMP.getdata.(updates_for_ins) == [[PointMass(1), PointMass(2)]]
        @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1)]

        run_autoupdate!(autoupdates_for_model)

        @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0)] # The marginal should not update at this point, because `y` has not been updated
        @test ReactiveMP.getdata.(updates_for_ins) == [[PointMass(1), PointMass(2)], [PointMass(2.0), PointMass(2.0)]]
        @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1)]

        update!(variable_y, 0)

        @test ReactiveMP.getdata.(marginals_θ) == [Beta(2.0, 2.0), Beta(2.0, 3.0)]
        @test ReactiveMP.getdata.(updates_for_ins) == [[PointMass(1), PointMass(2)], [PointMass(2.0), PointMass(2.0)]]
        @test ReactiveMP.getdata.(updates_for_y) == [PointMass(1), PointMass(0)]

        unsubscribe!([subscription_marginal_θ, subscription_updates_ins, subscription_updates_y])
    end
end

@testitem "`fetch` for `AutoUpdateMapping`" begin
    import RxInfer: AutoUpdateMapping, FetchRecentArgument

    @test @inferred(fetch(AutoUpdateMapping(+, (1, 2)))) === 3
    @test @inferred(fetch(AutoUpdateMapping(+, (1, AutoUpdateMapping(+, (1, 1)))))) === 3
    @test @inferred(fetch(AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), ([1], [2])))) == [3]

    stream_of_1 = FetchRecentArgument(:a, of(1))
    stream_of_2 = FetchRecentArgument(:b, of(2))

    @test @inferred(fetch(AutoUpdateMapping(+, (stream_of_1, stream_of_2)))) === 3
    @test @inferred(fetch(AutoUpdateMapping(+, (stream_of_1, AutoUpdateMapping(+, (1, stream_of_1)))))) === 3
    @test @inferred(fetch(AutoUpdateMapping(+, (stream_of_1, AutoUpdateMapping(+, (stream_of_1, 1)))))) === 3

    empty_stream_for_x = FetchRecentArgument(:x, RecentSubject(Int))
    empty_stream_for_y = FetchRecentArgument(:y, RecentSubject(Int))

    @test_throws "The initial value for `x` has not been specified, but is required in the `@autoupdates`." fetch(AutoUpdateMapping(+, (empty_stream_for_x, 1)))
    @test_throws "The initial value for `y` has not been specified, but is required in the `@autoupdates`." fetch(AutoUpdateMapping(+, (1, empty_stream_for_y)))

    stream_of_1_array = FetchRecentArgument(:a, of([1]))
    stream_of_2_array = FetchRecentArgument(:b, of([2]))

    @test @inferred(fetch(AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), (stream_of_1_array, stream_of_2_array)))) == [3]
    @test @inferred(fetch(AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), (stream_of_1_array, AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), (1, stream_of_1_array)))))) == [3]
    @test @inferred(fetch(AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), (stream_of_1_array, AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), (stream_of_1_array, 1)))))) == [3]

    @test @inferred(fetch(AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), (stream_of_1, stream_of_2_array)))) == [3]
    @test @inferred(fetch(AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), (stream_of_1_array, stream_of_2)))) == [3]
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
    @test autoupdate_argument_inexpr(:(1 .+ params.(q(x))))
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
    @test is_autoupdate_mapping_expr(:(params.(q(x))))
    @test is_autoupdate_mapping_expr(:(q(x) .+ 1))
    @test is_autoupdate_mapping_expr(:(f(q(x), q(x))))
    @test is_autoupdate_mapping_expr(:(f(q(x), μ(x))))
    @test is_autoupdate_mapping_expr(:(f(μ(x), q(x))))
    @test is_autoupdate_mapping_expr(:(f(μ(x), μ(x))))
end

@testitem "autoupdate_convert_mapping_expr" begin
    import RxInfer: autoupdate_convert_mapping_expr, AutoUpdateMapping, AutoUpdateFetchMarginalArgument, AutoUpdateFetchMessageArgument

    @test autoupdate_convert_mapping_expr(:(f(q(x)))) == :(RxInfer.AutoUpdateMapping(f, (RxInfer.AutoUpdateFetchMarginalArgument(:x),)))
    @test autoupdate_convert_mapping_expr(:(f.(q(x)))) == :(RxInfer.AutoUpdateMapping(Base.Broadcast.BroadcastFunction(f), (RxInfer.AutoUpdateFetchMarginalArgument(:x),)))
    @test autoupdate_convert_mapping_expr(:(q(x) .+ 1)) == :(RxInfer.AutoUpdateMapping(Base.Broadcast.BroadcastFunction(+), (RxInfer.AutoUpdateFetchMarginalArgument(:x), 1)))

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
