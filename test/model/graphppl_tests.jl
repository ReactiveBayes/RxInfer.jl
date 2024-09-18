@testitem "error_datavar_constvar_randomvar" begin
    import RxInfer: error_datavar_constvar_randomvar
    import GraphPPL: apply_pipeline
    import MacroTools: @capture

    input = :(a = randomvar())
    @test @capture(apply_pipeline(input, error_datavar_constvar_randomvar), error(_))

    input = :(a = datavar())
    @test @capture(apply_pipeline(input, error_datavar_constvar_randomvar), error(_))

    input = :(a = constvar())
    @test @capture(apply_pipeline(input, error_datavar_constvar_randomvar), error(_))

    input = :(x ~ Normal(0, 1))
    @test apply_pipeline(input, error_datavar_constvar_randomvar) == input
end

@testitem "compose_simple_operators_with_brackets pipeline" begin
    import RxInfer: compose_simple_operators_with_brackets
    import GraphPPL: apply_pipeline

    input = :(s ~ s1 + s2 + s3 + s4 + s5)
    output = :(s ~ (((s1 + s2) + s3) + s4) + s5)
    @test apply_pipeline(input, compose_simple_operators_with_brackets) == output

    input = :(s ~ (s1 + s2) + s3 + (s4 + s5))
    output = :(s ~ (((s1 + s2) + s3) + (s4 + s5)))
    @test apply_pipeline(input, compose_simple_operators_with_brackets) == output

    input = :(s ~ (s1 + s2) + (s3 + (s4 + s5)))
    output = :(s ~ (s1 + s2) + (s3 + (s4 + s5)))
    @test apply_pipeline(input, compose_simple_operators_with_brackets) == output

    input = :(s ~ Normal(μ = s1 + s2 + s3 + s4 + s5, 1.0))
    output = :(s ~ Normal(μ = (((s1 + s2) + s3) + s4) + s5, 1.0))
    @test apply_pipeline(input, compose_simple_operators_with_brackets) == output

    input = :(s ~ Normal(μ = s1 + s2 + s3 + s4 + s5, 1.0) where {a = 1})
    output = :(s ~ Normal(μ = (((s1 + s2) + s3) + s4) + s5, 1.0) where {a = 1})
    @test apply_pipeline(input, compose_simple_operators_with_brackets) == output

    # If not `~` should not change
    input = :(s = s1 + s2 + s3 + s4 + s5)
    output = :(s = s1 + s2 + s3 + s4 + s5)
    @test apply_pipeline(input, compose_simple_operators_with_brackets) == output

    input = :(s ~ s1 * s2 * s3 * s4 * s5)
    output = :(s ~ (((s1 * s2) * s3) * s4) * s5)
    @test apply_pipeline(input, compose_simple_operators_with_brackets) == output
end

@testitem "inject_tilderhs_aliases" begin
    import RxInfer: inject_tilderhs_aliases
    import GraphPPL: apply_pipeline
    import MacroTools: prettify

    input = :(a ~ b || c)
    output = :(a ~ ReactiveMP.OR(b, c))
    @test prettify(apply_pipeline(input, inject_tilderhs_aliases)) == prettify(output)

    input = :(b || c)
    output = :(b || c)
    @test prettify(apply_pipeline(input, inject_tilderhs_aliases)) == prettify(output)

    input = :(a ~ b && c)
    output = :(a ~ ReactiveMP.AND(b, c))
    @test prettify(apply_pipeline(input, inject_tilderhs_aliases)) == prettify(output)

    input = :(b && c)
    output = :(b && c)
    @test prettify(apply_pipeline(input, inject_tilderhs_aliases)) == prettify(output)

    input = :(a ~ b -> c)
    output = :(a ~ ReactiveMP.IMPLY(b, c))
    @test prettify(apply_pipeline(input, inject_tilderhs_aliases)) == prettify(output)

    input = :(a = b -> b + 1)
    output = :(a = b -> b + 1)
    @test prettify(apply_pipeline(input, inject_tilderhs_aliases)) == prettify(output)

    input = :(a ~ ¬b)
    output = :(a ~ ReactiveMP.NOT(b))
    @test prettify(apply_pipeline(input, inject_tilderhs_aliases)) == prettify(output)

    input = :(a ~ !b)
    output = :(a ~ ReactiveMP.NOT(b))
    @test prettify(apply_pipeline(input, inject_tilderhs_aliases)) == prettify(output)
end

@testitem "`@node` should properly define `GraphPPL` backend specific information" begin
    import RxInfer: ReactiveMPGraphPPLBackend
    import ReactiveMP: @node
    import GraphPPL
    using Static

    struct CustomStochasticNode end

    @node CustomStochasticNode Stochastic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

    backend = ReactiveMPGraphPPLBackend(Static.False())

    @test GraphPPL.NodeBehaviour(backend, CustomStochasticNode) === GraphPPL.Stochastic()
    @test GraphPPL.NodeType(backend, CustomStochasticNode) === GraphPPL.Atomic()
    @test GraphPPL.interfaces(backend, CustomStochasticNode, 4) === GraphPPL.StaticInterfaces((:out, :x, :y, :z))
    @test_throws ErrorException GraphPPL.interfaces(backend, CustomStochasticNode, 1)
    @test_throws ErrorException GraphPPL.interfaces(backend, CustomStochasticNode, 2)
    @test_throws ErrorException GraphPPL.interfaces(backend, CustomStochasticNode, 3)
    @test_throws ErrorException GraphPPL.interfaces(backend, CustomStochasticNode, 5)
    @test GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), CustomStochasticNode, (1, 2, 3)) === (x = 1, y = 2, z = 3)
    @test_throws ErrorException GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), CustomStochasticNode, (1, 2))
    @test_throws ErrorException GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), CustomStochasticNode, (1, 2, 3, 4))

    function f end

    @node typeof(f) Deterministic [out, in1, in2]

    @test GraphPPL.NodeBehaviour(backend, f) === GraphPPL.Deterministic()
    @test GraphPPL.NodeType(backend, f) === GraphPPL.Atomic()
    @test GraphPPL.interfaces(backend, f, 3) === GraphPPL.StaticInterfaces((:out, :in1, :in2))
    @test_throws ErrorException GraphPPL.interfaces(backend, f, 1)
    @test_throws ErrorException GraphPPL.interfaces(backend, f, 2)
    @test_throws ErrorException GraphPPL.interfaces(backend, f, 4)
    @test GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), f, (1, 2)) === (in1 = 1, in2 = 2)
    @test_throws ErrorException GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), f, (1,))
    @test_throws ErrorException GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), f, (1, 2, 3))
end

@testitem "`@node` should properly define `GraphPPL` backend specific information with node contraction allowed" begin
    import RxInfer: ReactiveMPGraphPPLBackend
    import ReactiveMP: @node
    import GraphPPL
    import Static

    struct CustomStochasticNode end

    @node CustomStochasticNode Stochastic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

    backend = ReactiveMPGraphPPLBackend(Static.True())

    @test GraphPPL.NodeBehaviour(backend, CustomStochasticNode) === GraphPPL.Stochastic()
    @test GraphPPL.NodeType(backend, CustomStochasticNode) === GraphPPL.Atomic()
    @test GraphPPL.interfaces(backend, CustomStochasticNode, 4) === GraphPPL.StaticInterfaces((:out, :x, :y, :z))
    @test_throws ErrorException GraphPPL.interfaces(backend, CustomStochasticNode, 1)
    @test_throws ErrorException GraphPPL.interfaces(backend, CustomStochasticNode, 2)
    @test_throws ErrorException GraphPPL.interfaces(backend, CustomStochasticNode, 3)
    @test_throws ErrorException GraphPPL.interfaces(backend, CustomStochasticNode, 5)
    @test GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), CustomStochasticNode, (1, 2, 3)) === (x = 1, y = 2, z = 3)
    @test_throws ErrorException GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), CustomStochasticNode, (1, 2))
    @test_throws ErrorException GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), CustomStochasticNode, (1, 2, 3, 4))

    function f end

    @node typeof(f) Deterministic [out, in1, in2]

    @test GraphPPL.NodeBehaviour(backend, f) === GraphPPL.Deterministic()
    @test GraphPPL.NodeType(backend, f) === GraphPPL.Atomic()
    @test GraphPPL.interfaces(backend, f, 3) === GraphPPL.StaticInterfaces((:out, :in1, :in2))
    @test_throws ErrorException GraphPPL.interfaces(backend, f, 1)
    @test_throws ErrorException GraphPPL.interfaces(backend, f, 2)
    @test_throws ErrorException GraphPPL.interfaces(backend, f, 4)
    @test GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), f, (1, 2)) === (in1 = 1, in2 = 2)
    @test_throws ErrorException GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), f, (1,))
    @test_throws ErrorException GraphPPL.default_parametrization(backend, GraphPPL.Atomic(), f, (1, 2, 3))
end
