@testitem "a removed or moved name says what replaces it" begin
    using RxInfer: removed_name_hint

    @test contains(
        removed_name_hint(:Probit), "`using ProbitMessagePassingRules`"
    )
    @test contains(removed_name_hint(:ARMeta), "`ARVMP(form, order, stype)`")
    @test contains(
        removed_name_hint(Symbol("@rule")), "`@define_message_update_rule`"
    )
    @test contains(removed_name_hint(:DeltaMeta), "`DeltaApproximation(")
    @test removed_name_hint(:certainly_not_a_name) === nothing

    # The hint reaches the `UndefVarError` a user sees.
    err = try
        Core.eval(Module(), :(ARMeta(1)))
    catch e
        e
    end
    @test err isa UndefVarError && contains(
        sprint(showerror, err), "Hint: `ARMeta` is from ReactiveMP v6"
    )
end

@testitem "infer says how to pass a model" begin
    @model function hint_model(y)
        y ~ Normal(mean = 0.0, precision = 1.0)
    end
    err = try
        infer(model = hint_model, data = (y = 1.0,))
    catch e
        e
    end
    @test err isa ErrorException
    @test contains(
        err.msg, "takes a model created by calling a `@model` function"
    ) && contains(err.msg, "call it with its arguments")
end

@testitem "a missing rule reaches the user as a RuleNotFoundError, with a pointer to the guide" begin
    @model function no_rule_model(y)
        μ ~ Normal(mean = 0.0, variance = 1.0)
        τ ~ Gamma(shape = 1.0, rate = 1.0)
        y ~ Normal(mean = μ, precision = τ)
    end
    err = @test_logs (
        :error, r"No message passing rule fits a node of the model"
    ) match_mode = :any try
        infer(
            model = no_rule_model(),
            data = (y = 1.0,),
            disable_inference_error_hint = true,
        )
    catch e
        e
    end
    @test err isa MessagePassingRulesBase.RuleNotFoundError
end
