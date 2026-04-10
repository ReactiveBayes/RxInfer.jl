@testitem "IID estimation trace to TensorBoard" begin
    using RxInfer, StableRNGs, TensorBoardLogger

    # A simple IID model: observations are drawn from a Normal with unknown mean and precision.
    # Mean-field constraints decouple q(μ) and q(τ) for variational inference.
    @model function iid_estimation(y)
        μ ~ Normal(mean = 0.0, precision = 0.1)
        τ ~ Gamma(shape = 1.0, rate = 1.0)
        y .~ Normal(mean = μ, precision = τ)
    end

    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end

    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    # Generate synthetic observations from a known distribution so the test is reproducible.
    hidden_μ = 3.1415
    hidden_τ = 2.7182
    dataset   = rand(StableRNG(42), NormalMeanPrecision(hidden_μ, hidden_τ), 25)

    # Run inference with `trace = true` so all internal events are recorded.
    # The trace is stored in the model metadata under the `:trace` key.
    results = infer(
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = 2,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    # Export the trace to TensorBoard format and verify the output.
    # `mktempdir` ensures the log files are written to a temporary directory
    # that is cleaned up automatically after the test, avoiding filesystem pollution.
    mktempdir() do log_dir
        output = RxInfer.convert_to_tensorboard(trace; output_file = log_dir)
        @test output == log_dir  # function returns the path it wrote to
        @test isdir(log_dir)     # directory was created
    end
end
