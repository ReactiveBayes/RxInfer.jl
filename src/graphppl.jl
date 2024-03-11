import GraphPPL
import ExponentialFamily

GraphPPL.factor_alias(::Type{Normal}, ::Val{(:mean, :var)}) = ExponentialFamily.NormalMeanVariance
GraphPPL.factor_alias(::Type{Normal}, ::Val{(:mean, :variance)}) = ExponentialFamily.NormalMeanVariance
GraphPPL.factor_alias(::Type{Normal}, ::Val{(:m, :v)}) = ExponentialFamily.NormalMeanVariance
GraphPPL.factor_alias(::Type{Normal}, ::Val{(:μ, :v)}) = ExponentialFamily.NormalMeanVariance

GraphPPL.factor_alias(::Type{MvNormal}, ::Val{(:mean, :covariance)}) = ExponentialFamily.MvNormalMeanCovariance
GraphPPL.factor_alias(::Type{MvNormal}, ::Val{(:μ, :Σ)}) = ExponentialFamily.MvNormalMeanCovariance

GraphPPL.factor_alias(::Type{MvNormal}, ::Val{(:mean, :precision)}) = ExponentialFamily.MvNormalMeanPrecision
GraphPPL.factor_alias(::Type{MvNormal}, ::Val{(:μ, :Λ)}) = ExponentialFamily.MvNormalMeanPrecision