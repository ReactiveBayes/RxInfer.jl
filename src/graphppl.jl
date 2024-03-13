import GraphPPL
import ExponentialFamily

GraphPPL.factor_alias(::Type{Normal}, ::Val{(:μ, :v)}) = ExponentialFamily.NormalMeanVariance
GraphPPL.factor_alias(::Type{Normal}, ::Val{(:μ, :τ)}) = ExponentialFamily.NormalMeanPrecision

GraphPPL.interfaces(::Type{<:ExponentialFamily.NormalMeanVariance}, _) = GraphPPL.StaticInterfaces((:out, :μ, :v))
GraphPPL.interfaces(::Type{<:ExponentialFamily.NormalMeanPrecision}, _) = GraphPPL.StaticInterfaces((:out, :μ, :τ))

GraphPPL.interface_aliases(::Type{Normal}) = GraphPPL.StaticInterfaceAliases((
    (:mean, :μ), (:m, :μ), (:variance, :v), (:var, :v), (:τ⁻¹, :v), (:σ², :v), (:precision, :τ), (:prec, :τ), (:p, :τ), (:w, :τ), (:σ⁻², :τ), (:γ, :τ)
))

GraphPPL.factor_alias(::Type{MvNormal}, ::Val{(:μ, :Σ)}) = ExponentialFamily.MvNormalMeanCovariance
GraphPPL.factor_alias(::Type{MvNormal}, ::Val{(:μ, :Λ)}) = ExponentialFamily.MvNormalMeanPrecision

GraphPPL.interface_aliases(::Type{MvNormal}) =
    GraphPPL.StaticInterfaceAliases(((:mean, :μ), (:m, :μ), (:covariance, :Σ), (:cov, :Σ), (:Λ⁻¹, :Σ), (:V, :Σ), (:precision, :Λ), (:prec, :Λ), (:W, :Λ), (:Σ⁻¹, :Λ)))

GraphPPL.factor_alias(::Type{Gamma}, ::Val{(:α, :θ)}) = ExponentialFamily.GammaShapeScale
GraphPPL.factor_alias(::Type{Gamma}, ::Val{(:α, :β)}) = ExponentialFamily.GammaShapeRate

GraphPPL.interfaces(::Type{<:ExponentialFamily.GammaShapeScale}, _) = GraphPPL.StaticInterfaces((:out, :α, :θ))
GraphPPL.interfaces(::Type{<:ExponentialFamily.GammaShapeRate}, _) = GraphPPL.StaticInterfaces((:out, :α, :β))

GraphPPL.interface_aliases(::Type{Gamma}) = GraphPPL.StaticInterfaceAliases(((:a, :α), (:shape, :α), (:β⁻¹, :θ), (:scale, :θ), (:θ⁻¹, :β), (:rate, :β)))

GraphPPL.NodeBehaviour(::Type{ReactiveMP.AR}) = GraphPPL.Stochastic()