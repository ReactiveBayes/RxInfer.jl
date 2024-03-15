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