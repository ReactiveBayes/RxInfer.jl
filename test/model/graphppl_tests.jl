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