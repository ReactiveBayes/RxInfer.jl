module RxInferNodeTest

using Test
using RxInfer
using Random

@testset "@node macro integration tests" begin
    @testset "make_node compatibility tests for stochastic nodes" begin
        struct CustomStochasticNode end

        @node CustomStochasticNode Stochastic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

        @test sdtype(CustomStochasticNode) === Stochastic()

        cx = constvar(:cx, 1.0)
        cy = constvar(:cy, 1.0)
        cz = constvar(:cz, 1.0)

        model = FactorGraphModel()

        snode, svar = make_node(model, FactorNodeCreationOptions(MeanField(), nothing, nothing), CustomStochasticNode, AutoVar(:cout), cx, cy, cz)

        @test snode ∈ getnodes(model)
        @test svar ∈ getrandom(model)

        @test snode !== nothing
        @test typeof(svar) <: RandomVariable
        @test factorisation(snode) === ((1,), (2,), (3,), (4,))
    end

    @testset "make_node compatibility tests for deterministic nodes" begin
        struct CustomDeterministicNode end

        CustomDeterministicNode(x, y, z) = x + y + z

        @node CustomDeterministicNode Deterministic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

        @test sdtype(CustomDeterministicNode) === Deterministic()

        cx = constvar(:cx, 1.0)
        cy = constvar(:cy, 1.0)
        cz = constvar(:cz, 1.0)

        model = FactorGraphModel()

        snode, svar = make_node(model, FactorNodeCreationOptions(MeanField(), nothing, nothing), CustomDeterministicNode, AutoVar(:cout), cx, cy, cz)

        @test svar ∈ getconstant(model)

        @test snode === nothing
        @test typeof(svar) <: ConstVariable
    end

    @testset "`FactorGraphModel` compatibility/correctness with functional dependencies pipelines" begin
        import ReactiveMP: activate!

        struct DummyStochasticNode end

        @node DummyStochasticNode Stochastic [x, y, z]

        function make_dummy_model(factorisation, pipeline)
            m = FactorGraphModel()
            x = randomvar(m, :x)
            y = randomvar(m, :y)
            z = randomvar(m, :z)
            make_node(m, FactorNodeCreationOptions(nothing, nothing, nothing), Uninformative, x)
            make_node(m, FactorNodeCreationOptions(nothing, nothing, nothing), Uninformative, y)
            make_node(m, FactorNodeCreationOptions(nothing, nothing, nothing), Uninformative, z)
            node = make_node(m, FactorNodeCreationOptions(factorisation, nothing, pipeline), DummyStochasticNode, x, y, z)
            activate!(m)
            return m, x, y, z, node
        end

        @testset "Default functional dependencies" begin
            import ReactiveMP: DefaultFunctionalDependencies

            @testset "Default functional dependencies: FullFactorisation" begin
                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 2 && name(x_msgdeps[1]) === :y && name(x_msgdeps[2]) === :z
                @test length(x_mgdeps) === 0

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 0

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y
                @test length(z_mgdeps) === 0
            end

            @testset "Default functional dependencies: MeanField" begin
                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :y && name(x_mgdeps[2]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y
            end

            @testset "Default functional dependencies: Structured factorisation" begin
                # We test `(x, y), (z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 2), (3,)), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :y
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :x
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x_y

                ## --- ##

                # We test `(x, ), (y, z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1,), (2, 3)), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :z
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :y
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x

                ## --- ##

                # We test `(x, z), (y, )` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 3), (2,)), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x_z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :x
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :y
            end
        end

        @testset "Require inbound message functional dependencies" begin
            import ReactiveMP: RequireMessageFunctionalDependencies

            @testset "Require inbound message functional dependencies: FullFactorisation" begin
                # Require inbound message on `x`
                pipeline = RequireMessageFunctionalDependencies((1,), (NormalMeanVariance(0.123, 0.123),))

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 0
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(x_msgdeps[1]))) == (0.123, 0.123)

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 0

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y
                @test length(z_mgdeps) === 0

                ## -- ## 

                # Require inbound message on `y` and `z`
                pipeline = RequireMessageFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 2 && name(x_msgdeps[1]) === :y && name(x_msgdeps[2]) === :z
                @test length(x_mgdeps) === 0

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 0
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[2]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 0
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[3])))
            end

            @testset "Require inbound message functional dependencies: MeanField" begin
                # Require inbound message on `x`
                pipeline = RequireMessageFunctionalDependencies((1,), (NormalMeanVariance(0.123, 0.123),))

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :x
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :y && name(x_mgdeps[2]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(x_msgdeps[1]))) == (0.123, 0.123)

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y

                ## -- ## 

                # Require inbound message on `y` and `z`
                pipeline = RequireMessageFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :y && name(x_mgdeps[2]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :y
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[1]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :z
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[1])))
            end

            @testset "Require inbound message dependencies: Structured factorisation" begin
                # Require inbound message on `y` and `z`
                pipeline = RequireMessageFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, y), (z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 2), (3,)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :y
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[2]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :z
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x_y
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[1])))

                ## --- ##

                # Require inbound message on `y` and `z`
                pipeline = RequireMessageFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, ), (y, z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1,), (2, 3)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :y && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[1]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :y && name(z_msgdeps[2]) === :z
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[2])))

                ## --- ##

                # Require inbound message on `y` and `z`
                pipeline = RequireMessageFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, z), (y, )` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 3), (2,)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :y
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x_z
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[1]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :z
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :y
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[2])))
            end
        end

        @testset "Require marginal functional dependencies" begin
            import ReactiveMP: RequireMarginalFunctionalDependencies

            @testset "Require marginal functional dependencies: FullFactorisation" begin
                # Require marginal on `x`
                pipeline = RequireMarginalFunctionalDependencies((1,), (NormalMeanVariance(0.123, 0.123),))

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 2 && name(x_msgdeps[1]) === :y && name(x_msgdeps[2]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :x
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(x, IncludeAll()))) == (0.123, 0.123)

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 0

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y
                @test length(z_mgdeps) === 0

                ## -- ## 

                # Require marginals on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 2 && name(x_msgdeps[1]) === :y && name(x_msgdeps[2]) === :z
                @test length(x_mgdeps) === 0

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :y
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))
            end

            @testset "Require marginal functional dependencies: MeanField" begin
                # Require marginal on `x`
                pipeline = RequireMarginalFunctionalDependencies((1,), (NormalMeanVariance(0.123, 0.123),))

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 3 && name(x_mgdeps[1]) === :x && name(x_mgdeps[2]) === :y && name(x_mgdeps[3]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(x, IncludeAll()))) == (0.123, 0.123)

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y

                ## -- ## 

                # Require marginals on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :y && name(x_mgdeps[2]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 3 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :y && name(y_mgdeps[3]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 3 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y && name(z_mgdeps[3]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))
            end

            @testset "Require marginal functional dependencies: Structured factorisation" begin
                # Require marginal on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, y), (z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 2), (3,)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :y
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :x
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :y && name(y_mgdeps[2]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x_y && name(z_mgdeps[2]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))

                ## --- ##

                # Require marginals on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, ), (y, z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1,), (2, 3)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :z
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :y
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :y
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))

                ## --- ##

                # Require marginals on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, z), (y, )` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 3), (2,)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x_z && name(y_mgdeps[2]) === :y
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :x
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :y && name(z_mgdeps[2]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))
            end
        end

        @testset "Require everything functional dependencies" begin
            import ReactiveMP: RequireEverythingFunctionalDependencies

            @testset "Require everything functional dependencies: FullFactorisation" begin
                pipeline = RequireEverythingFunctionalDependencies()

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :x_y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x_y_z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x_y_z
            end

            @testset "Require everything functional dependencies: MeanField" begin
                pipeline = RequireEverythingFunctionalDependencies()

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 3 && name(x_mgdeps[1]) === :x && name(x_mgdeps[2]) === :y && name(x_mgdeps[3]) === :z
                @test length(x_mgdeps) === 3 && name(x_mgdeps[1]) === :x && name(x_mgdeps[2]) === :y && name(x_mgdeps[3]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 3 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :y && name(y_mgdeps[3]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 3 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y && name(z_mgdeps[3]) === :z
            end

            @testset "Require everything dependencies: Structured factorisation" begin
                pipeline = RequireEverythingFunctionalDependencies()

                # We test `(x, y), (z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 2), (3,)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :x_y && name(x_mgdeps[2]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x_y && name(y_mgdeps[2]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x_y && name(z_mgdeps[2]) === :z

                ## --- ##

                pipeline = RequireEverythingFunctionalDependencies()

                # We test `(x, ), (y, z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1,), (2, 3)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :x && name(x_mgdeps[2]) === :y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :y_z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y_z

                ## --- ##

                pipeline = RequireEverythingFunctionalDependencies()

                # We test `(x, z), (y, )` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 3), (2,)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x)

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :x_z && name(x_mgdeps[2]) === :y

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y)

                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x_z && name(y_mgdeps[2]) === :y

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)

                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x_z && name(z_mgdeps[2]) === :y
            end
        end
    end
end

end
