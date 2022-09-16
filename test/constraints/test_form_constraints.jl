module RxInferFormConstraintsSpecificationTest

using Test, Logging
using RxInfer

import RxInfer: PointMassFormConstraint, SampleListFormConstraint, FixedMarginalFormConstraint
import ReactiveMP: CompositeFormConstraint
import ReactiveMP: resolve_marginal_form_prod, resolve_messages_form_prod
import ReactiveMP: activate!

@testset "Form constraints specification" begin
    @testset "Use case #1" begin
        cs = @constraints begin
            q(x)::PointMass
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #2" begin
        cs = @constraints begin
            q(x)::Nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            form === UnspecifiedFormConstraint() && prod === ProdAnalytical()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #2" begin
        cs = @constraints begin
            q(x)::SampleList(5000, LeftProposal())
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            typeof(form) <: SampleListFormConstraint && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #3" begin
        cs = @constraints begin
            q(x)::PointMass(optimizer = "optimizer")
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            typeof(form) <: PointMassFormConstraint && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #4" begin
        cs = @constraints begin
            q(x)::SampleList(5000, LeftProposal())::PointMass(optimizer = "optimizer")
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            typeof(form) <: CompositeFormConstraint && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #5" begin
        @constraints function cs5(flag)
            if flag
                q(x)::PointMass
            end
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(true), :x)
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(true), :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(false), :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(false), :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(true), :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(false), :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(true), :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(false), :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #6" begin
        cs = @constraints begin
            μ(x)::PointMass
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #7" begin
        cs = @constraints begin
            μ(x)::Nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            form === UnspecifiedFormConstraint() && prod === ProdAnalytical()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #8" begin
        cs = @constraints begin
            μ(x)::SampleList(5000, LeftProposal())
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            typeof(form) <: SampleListFormConstraint && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #9" begin
        cs = @constraints begin
            μ(x)::PointMass(optimizer = "optimizer")
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            typeof(form) <: PointMassFormConstraint && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #10" begin
        cs = @constraints begin
            μ(x)::SampleList(5000, LeftProposal())::PointMass(optimizer = "optimizer")
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            typeof(form) <: CompositeFormConstraint && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #11" begin
        @constraints function cs5(flag)
            if flag
                μ(x)::PointMass
            end
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(true), :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(true), :x)
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(false), :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(false), :x)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(true), :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(true), :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(false), :y)
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(false), :y)
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #12" begin
        cs = @constraints begin
            q(x)::PointMass
            μ(x)::SampleList(5000, LeftProposal())
            q(y)::Nothing
            μ(y)::PointMass
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :x)
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :x)
            typeof(form) <: SampleListFormConstraint && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, :y)
            form === UnspecifiedFormConstraint() && prod === ProdAnalytical()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, :y)
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end
    end

    @testset "Warning case #1" begin
        model = FactorGraphModel()

        cs_with_warn = @constraints [warn = true] begin
            q(x)::PointMass # Unknown variable for marginal
        end

        cs_without_warn = @constraints [warn = false] begin
            q(x)::PointMass # Unknown variable for marginal
        end

        @test_logs (:warn, r".*q(.*).*no random variable") activate!(cs_with_warn, getnodes(model), getvariables(model))
        @test_logs min_level = Logging.Warn activate!(cs_without_warn, getnodes(model), getvariables(model))
    end

    @testset "Warning case #2" begin
        model = FactorGraphModel()

        cs_with_warn = @constraints [warn = true] begin
            μ(x)::PointMass # Unknown variable for marginal
        end

        cs_without_warn = @constraints [warn = false] begin
            μ(x)::PointMass # Unknown variable for marginal
        end

        @test_logs (:warn, r".*μ(.*).*no random variable") activate!(cs_with_warn, getnodes(model), getvariables(model))
        @test_logs min_level = Logging.Warn activate!(cs_without_warn, getnodes(model), getvariables(model))
    end

    @testset "Warning case #3" begin
        model = FactorGraphModel()

        x = datavar(model, :x, Float64)

        cs_with_warn = @constraints [warn = true] begin
            q(x)::PointMass # Unknown variable for marginal
        end

        cs_without_warn = @constraints [warn = false] begin
            q(x)::PointMass # Unknown variable for marginal
        end

        @test_logs (:warn, r".*q(.*).*is not a random variable") activate!(cs_with_warn, getnodes(model), getvariables(model))
        @test_logs min_level = Logging.Warn activate!(cs_without_warn, getnodes(model), getvariables(model))
    end

    @testset "Warning case #4" begin
        model = FactorGraphModel()

        x = constvar(model, :x, 1.0)

        cs_with_warn = @constraints [warn = true] begin
            q(x)::PointMass # Unknown variable for marginal
        end

        cs_without_warn = @constraints [warn = false] begin
            q(x)::PointMass # Unknown variable for marginal
        end

        @test_logs (:warn, r".*q(.*).*is not a random variable") activate!(cs_with_warn, getnodes(model), getvariables(model))
        @test_logs min_level = Logging.Warn activate!(cs_without_warn, getnodes(model), getvariables(model))
    end

    @testset "Error case #1" begin
        @test_throws ErrorException @constraints begin
            q(x)::Nothing
            q(x)::PointMass
        end

        @test_throws ErrorException @constraints begin
            μ(x)::Nothing
            μ(x)::PointMass
        end
    end
end

end
