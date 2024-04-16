# [Built-in Functional Forms](@id lib-forms)

This section describes built-in functional forms that can be used for posterior marginal and/or messages form constraints specification. Read more information about constraints specification syntax in the [Constraints Specification](@ref user-guide-constraints-specification) section.

## [Background](@id lib-forms-background)

In message passing framework, in order to compute a posterior over some latent state $q(x)$, it is necessary to compute a normalized product of two messages
$$q(x) = \frac{\mu_1(x) \mu_2(x)}{\int \mu_1(x) \mu_2(x) \mathrm{d}x}.$$
In some situations, when functional forms of $\mu_1(x)$ and $\mu_2(x)$ are know in advance, it is possible to compute the normalized product efficiently and analytically. It is, however, not always the case, since the messages can have arbitrary functional form and it is not always easy to compute the normalization factor.

Functional forms help to circumvent this. They implement a custom callback on the product of two messages, which cannot 
be computed analytically. Essentially, a functional form constraint defines a functional `F`, such that 
$$q(x) = F[\mu_1, \mu_2] \approx \frac{\mu_1(x) \mu_2(x)}{\int \mu_1(x) \mu_2(x) \mathrm{d}x}.$$

See also [Bethe Free Energy](@ref lib-bethe-free-energy) section for more information on variational inference and posterior computation.

## [UnspecifiedFormConstraint](@id lib-forms-unspecified-constraint)

Unspecified functional form constraint is used by default and uses only analytical update rules for computing posterior marginals. Throws an error if a product of two colliding messages cannot be computed analytically.

```@example constraints-functional-forms
using RxInfer, Distributions #hide
@constraints begin 
    # This is the default setting for all latent variables
    q(x) :: UnspecifiedFormConstraint() 
end
nothing #hide
```

## [PointMassFormConstraint](@id lib-forms-point-mass-constraint)

The most basic form of posterior marginal approximation is the `PointMass` function. In a few words `PointMass` represents the delta function. In the context of functional form constraints `PointMass` approximation corresponds to the MAP estimate. For a given distribution `d` - `PointMass` functional form simply finds the `argmax` of the `logpdf` of `q(x)`, thus $q(x) = F[\mu_1, \mu_2] = \delta(x - \arg\min_{x} \mu_1(x) \mu_2(x))$. This is especially useful when exact functional form of `q(x)` is not available or cannot be parametrized efficiently. 

```@example constraints-functional-forms
@constraints begin 
    q(x) :: PointMassFormConstraint()
end

@constraints begin 
    q(x) :: PointMassFormConstraint(starting_point = (args...) -> 1.0)
end
nothing #hide
```

```@docs 
RxInfer.PointMassFormConstraint
RxInfer.default_point_mass_form_constraint_optimizer
RxInfer.default_point_mass_form_constraint_starting_point
RxInfer.default_point_mass_form_constraint_boundaries
```

## [SampleListFormConstraint](@id lib-forms-sample-list-constraint)

`SampleListFormConstraints` approximates the resulting posterior marginal (product of two colliding messages) as a list of weighted samples. Hence, it requires one of the arguments to be a proper distribution (or at least the inference backend should be able to sample from it). This setting is controlled with `LeftProposal()`, `RightProposal()` or `AutoProposal()` objects. It also accepts an optional `method` object, but the only one available sampling method currently is the `BayesBase.BootstrapImportanceSampling`.

```@example constraints-functional-forms
@constraints begin 
    q(x) :: SampleListFormConstraint(1000)
    # or 
    q(y) :: SampleListFormConstraint(1000, LeftProposal())
end
nothing #hide
```

```@docs 
RxInfer.SampleListFormConstraint
RxInfer.AutoProposal
RxInfer.LeftProposal
RxInfer.RightProposal
```

## [FixedMarginalFormConstraint](@id lib-forms-fixed-marginal-constraint)

Fixed marginal form constraint replaces the resulting posterior marginal obtained during the inference procedure with the prespecified one. Worth to note that the inference backend still tries to compute real posterior marginal and may fail during this process. Might be useful for debugging purposes. If `nothing` is passed then the computed posterior marginal is returned (see also [`UnspecifiedFormConstraint`](@ref lib-forms-unspecified-constraint)).

```@example constraints-functional-forms
@constraints function block_updates(x_posterior = nothing) 
    # `nothing` returns the computed posterior marginal
    q(x) :: FixedMarginalFormConstraint(x_posterior)
end
nothing #hide
```

```@docs 
RxInfer.FixedMarginalFormConstraint
```

It is also possible to control the constraint manually, e.g:

```@example constraints-functional-forms
form_constraint = FixedMarginalFormConstraint(nothing)

constraints_specification = @constraints function manual_block_updates(form_constraint) 
    q(x) :: form_constraint
end

# later on ...
form_constraint.fixed_value = Gamma(1.0, 1.0)
```

## [CompositeFormConstraint](@id lib-forms-composite-constraint)

It is possible to create a composite functional form by stacking operators, e.g:

```@example constraints-functional-forms
@constraints begin 
    q(x) :: SampleListFormConstraint(1000) :: PointMassFormConstraint()
end
```

## [Custom functional forms](@id lib-forms-custom-constraints)

See the [ReactiveMP.jl library documentation](https://reactivebayes.github.io/ReactiveMP.jl/stable/) for more information about defining novel custom functional forms that are compatible with `ReactiveMP` inference backend.