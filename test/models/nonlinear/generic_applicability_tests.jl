@testitem "Nonlinear models: single input - single output" begin
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    # As a bonus we test that the function can depend on a global variable
    # A particular value does not matter here, only the fact that it runs
    globalvar = 0

    function f₁(x)
        return sqrt.(x .+ globalvar)
    end

    function f₁_inv(x)
        return x .^ 2
    end

    @model function delta_1input(y, meta)
        c = zeros(2)
        c[1] = 1.0
        x ~ MvNormal(μ = ones(2), Λ = diageye(2))
        z := f₁(x) where {meta = meta}
        θ ~ Normal(μ = dot(z, c), σ² = 1.0)
        y ~ Normal(μ = θ, σ² = 0.5)
    end

    # We test here different approximation methods
    metas = (
        DeltaMeta(method = Linearization(), inverse = f₁_inv),
        DeltaMeta(method = Unscented(), inverse = f₁_inv),
        DeltaMeta(method = Linearization()),
        DeltaMeta(method = Unscented()),
        Linearization(),
        Unscented()
    )

    results = map(metas) do meta
        return infer(model = delta_1input(meta = meta), data = (y = 1.0,), free_energy = true, iterations = 10)
    end

    @test all(result -> result isa RxInfer.InferenceResult, results)
    @test all(result -> all(<=(0), diff(result.free_energy)), results)
end

@testitem "Nonlinear models: multiple input (x2) - single output" begin
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    function f₂(x, θ)
        return x .+ θ
    end

    function f₂_x(θ, z)
        return z .- θ
    end

    function f₂_θ(x, z)
        return z .- x
    end

    @model function delta_2inputs(meta, y)
        c = zeros(2)
        c[1] = 1.0

        θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
        x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
        z := f₂(x, θ) where {meta = meta}
        w ~ Normal(μ = dot(z, c), σ² = 1.0)
        y ~ Normal(μ = w, σ² = 0.5)
    end

    metas = (
        DeltaMeta(method = Linearization(), inverse = (f₂_x, f₂_θ)),
        DeltaMeta(method = Unscented(), inverse = (f₂_x, f₂_θ)),
        DeltaMeta(method = Linearization()),
        DeltaMeta(method = Unscented()),
        Linearization(),
        Unscented()
    )

    results = map(metas) do meta
        return infer(model = delta_2inputs(meta = meta), data = (y = 1.0,), free_energy = true, iterations = 10)
    end

    @test all(result -> result isa RxInfer.InferenceResult, results)
    @test all(result -> all(<=(0), diff(result.free_energy)), results)
end

@testitem "Nonlinear models: multiple input (x3) - single output" begin
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    function f₃(x, θ, ζ)
        return x .+ θ .+ ζ
    end

    @model function delta_3inputs(meta, y)
        c = zeros(2)
        c[1] = 1.0

        θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
        ζ ~ MvNormal(μ = 0.5ones(2), Λ = diageye(2))
        x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
        z := f₃(x, θ, ζ) where {meta = meta}
        w ~ Normal(μ = dot(z, c), σ² = 1.0)
        y ~ Normal(μ = w, σ² = 0.5)
    end

    metas = (DeltaMeta(method = Linearization()), DeltaMeta(method = Unscented()), Linearization(), Unscented())

    results = map(metas) do meta
        return infer(model = delta_3inputs(meta = meta), data = (y = 1.0,), free_energy = true, iterations = 10)
    end

    @test all(result -> result isa RxInfer.InferenceResult, results)
    @test all(result -> all(<=(0), diff(result.free_energy)), results)
end

@testitem "Nonlinear models: multiple inputs (Multivariate x Univariate) - single output (Multivariate)" begin
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    function f₄(x, θ)
        return θ .* x
    end

    @model function delta_2input_1d2d(meta, y)
        c = zeros(2)
        c[1] = 1.0

        θ ~ Normal(μ = 0.5, γ = 1.0)
        x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
        z := f₄(x, θ) where {meta = meta}
        w ~ Normal(μ = dot(z, c), σ² = 1.0)
        y ~ Normal(μ = w, σ² = 0.5)
    end

    metas = (DeltaMeta(method = Linearization()), DeltaMeta(method = Unscented()), Linearization(), Unscented())

    results = map(metas) do meta
        return infer(model = delta_2input_1d2d(meta = meta), data = (y = 1.0,), free_energy = true, iterations = 10)
    end

    @test all(result -> result isa RxInfer.InferenceResult, results)
    @test all(result -> all(<=(0), diff(result.free_energy)), results)
end

@testitem "Nonlinear models: single input - multiple output" begin
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    g(x, z) = x .* z

    @meta function test_meta()
        g() -> Linearization()
    end

    # Model Creation
    @model function test_model(z, y)
        x ~ NormalMeanVariance(1.0, 1.0)
        u := g(x, z)
        y ~ MvNormalMeanPrecision(u, diageye(2))

    end

    results = infer(
        model = test_model(),
        data = (z = [1, 2], y = [1, 2]),
        meta = test_meta(),
    )

    @test results isa RxInfer.InferenceResult

end