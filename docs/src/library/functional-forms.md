# [Built-in Functional Forms](@id lib-forms)

This section describes built-in functional forms that can be used for posterior marginal and/or messages form constraints specification. Read more information about constraints specification syntax in the [Constraints Specification](@ref user-guide-constraints-specification) section.

## [Custom functional forms](@id lib-forms-custom-constraints)

See the [ReactiveMP.jl library documentation](https://biaslab.github.io/ReactiveMP.jl/stable/) for more information about defining novel custom functional forms that are compatible with `ReactiveMP` inference backend.

## [UnspecifiedFormConstraint](@id lib-forms-unspecified-constraint)

Unspecified functional form constraint is used by default and uses only analytical update rules for computing posterior marginals. Throws an error if a product of two colliding messages cannot be computed analytically.

```julia
@constraints begin 
    q(x) :: Nothing # This is the default setting
end
```

## [PointMassFormConstraint](@id lib-forms-point-mass-constraint)

The most basic form of posterior marginal approximation is the `PointMass` function. In a few words `PointMass` represents delta function. In the context of functional form constraints `PointMass` approximation corresponds to the MAP estimate. For a given distribution `d` - `PointMass` functional form simply finds the `argmax` of the `logpdf(d, x)` by default. 

```julia
@constraints begin 
    q(x) :: PointMass # Materializes to the `PointMassFormConstraint` object
end
```

```@docs 
RxInfer.PointMassFormConstraint
```

## [SampleListFormConstraint](@id lib-forms-sample-list-constraint)

`SampleListFormConstraints` approximates the resulting posterior marginal (product of two colliding messages) as a list of weighted samples. Hence, it requires one of the arguments to be a proper distribution (or at least be able to sample from it). This setting is controlled with `LeftProposal()`, `RightProposal()` or `AutoProposal()` objects. It also accepts an optional `method` object, but the only one available sampling method currently is the `BootstrapImportanceSampling`.

```julia
@constraints begin 
    q(x) :: SampleList(1000)
    # or 
    q(y) :: SampleList(1000, LeftProposal())
end
```

```@docs 
RxInfer.SampleListFormConstraint
```

## [FixedMarginalFormConstraint](@id lib-forms-fixed-marginal-constraint)

Fixed marginal form constraint replaces the resulting posterior marginal obtained during the inference procedure with the prespecified one. Worth to note that the inference backend still tries to compute real posterior marginal and may fail during this process. Might be useful for debugging purposes. If `nothing` is passed then the computed posterior marginal is returned.

```julia
@constraints function block_updates(x_posterior = nothing) 
    # `nothing` returns the computed posterior marginal
    q(x) :: Marginal(x_posterior)
end
```

```@docs 
RxInfer.FixedMarginalFormConstraint
```

## [CompositeFormConstraint](@id lib-forms-composite-constraint)

It is possible to create a composite functional form constraint with either `+` operator or using `@constraints` macro, e.g:

```julia
form_constraint = SampleListFormConstraint(1000) + PointMassFormConstraint()
```

```julia
@constraints begin 
    q(x) :: SampleList(1000) :: PointMass()
end
```