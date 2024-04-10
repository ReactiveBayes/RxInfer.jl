@testitem "Default postprocessing" begin
    
    # Default postprocessing step removes Marginal type wrapper if no addons are present, 
    # and keeps the Marginal type wrapper otherwise
    @test inference_postprocess(DefaultPostprocess(), Marginal(1.0, false, false, nothing)) == 1.0
    @test inference_postprocess(DefaultPostprocess(), Marginal(1.0, false, false, 1)) == Marginal(1.0, false, false, 1)

end

@testitem "UnpackMarginal postprocessing" begin
    @test inference_postprocess(UnpackMarginalPostprocess(), Marginal(1.0, false, false, nothing)) == 1.0
    @test inference_postprocess(UnpackMarginalPostprocess(), Marginal(1.0, false, false, 1)) == 1.0
end

@testitem "Noop postprocessing" begin
    @test inference_postprocess(NoopPostprocess(), Marginal(1.0, false, false, nothing)) == Marginal(1.0, false, false, nothing)
    @test inference_postprocess(NoopPostprocess(), Marginal(1.0, false, false, 1)) == Marginal(1.0, false, false, 1)
end

@testitem "Custom postprocessing" begin
    struct CustomPostprocess end

    RxInfer.inference_postprocess(::CustomPostprocess, result::Marginal) = string(ReactiveMP.getdata(result))

    @model function beta_bernoulli(y)
        θ ~ Beta(1, 1)
        y ~ Bernoulli(θ)
    end

    result = infer(
        model = beta_bernoulli(),
        data  = (y = 1.,),
        postprocess = CustomPostprocess()
    )

    @test occursin("Beta{Float64}(α=2.0, β=1.0)", result.posteriors[:θ])
end