import GraphPPL
import ExponentialFamily

GraphPPL.factor_alias(::Type{MvNormal}, ::Val{(:mean, :covariance)}) = ExponentialFamily.MvNormalMeanCovariance
GraphPPL.factor_alias(::Type{MvNormal}, ::Val{(:μ, :Σ)}) = ExponentialFamily.MvNormalMeanCovariance