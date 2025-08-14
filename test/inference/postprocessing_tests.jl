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

    result = infer(model = beta_bernoulli(), data = (y = 1.0,), postprocess = CustomPostprocess())

    @test occursin("Beta{Float64}(α=2.0, β=1.0)", result.posteriors[:θ])
end

@testitem "Postprocessing should not be invoked when error occurs immediately" begin

    # We are going to throw in the rule
    struct MyCustomNode end

    @node MyCustomNode Stochastic [out, in]

    @rule MyCustomNode(:out, Marginalisation) (q_in::Any,) = begin
        throw(ErrorException("This is a test error"))
    end

    struct CustomPostprocessShouldNotBeInvoked end

    RxInfer.inference_postprocess(::CustomPostprocessShouldNotBeInvoked, result::Any) = error("This should not be invoked")

    @model function my_model_with_error(y)
        θ ~ MyCustomNode(1)
        y ~ Bernoulli(θ)
    end

    result = infer(model = my_model_with_error(), data = (y = 1.0,), postprocess = CustomPostprocessShouldNotBeInvoked(), catch_exception = true)

    @test result.error[1] isa ErrorException
    @test result.error[1].msg == "This is a test error"
    @test result.posteriors[:θ] === missing
end
