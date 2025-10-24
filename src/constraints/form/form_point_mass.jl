export PointMassFormConstraint

import ReactiveMP: default_form_check_strategy, default_prod_constraint, constrain_form
import DomainSets: Domain, infimum, supremum
import Optim

using BayesBase, Distributions, ExponentialFamily
using SparseArrays

"""
    PointMassFormConstraint

One of the form constraint objects. Constraint a message to be in a form of dirac's delta point mass. 
By default uses `Optim.jl` package to find argmin of `-logpdf(x)`. 
Accepts custom `optimizer` callback which might be used to customise optimisation procedure with different packages 
or different arguments for `Optim.jl` package.

# Keyword arguments
- `optimizer`: specifies a callback function for logpdf optimisation. See also: `RxInfer.default_point_mass_form_constraint_optimizer`
- `starting_point`: specifies a callback function for initial optimisation point: See also: `RxInfer.default_point_mass_form_constraint_starting_point`
- `boundaries`: specifies a callback function for determining optimisation boundaries: See also: `RxInfer.default_point_mass_form_constraint_boundaries`

## Custom optimizer callback interface

```julia
# This is an example of the `custom_optimizer` interface
function custom_optimizer(::Type{ Univariate }, ::Type{ Continuous }, constraint::PointMassFormConstraint, distribution)
    # should return argmin of the -logpdf(distribution)
end
```

## Custom starting point callback interface

```julia
# This is an example of the `custom_starting_point` interface
function custom_starting_point(::Type{ Univariate }, ::Type{ Continuous }, constraint::PointMassFormConstraint, distribution)
    # built-in optimizer expects an array, even for a univariate distribution
    return [ 0.0 ] 
end
```

## Custom boundaries callback interface

```julia
# This is an example of the `custom_boundaries` interface
function custom_boundaries(::Type{ Univariate }, ::Type{ Continuous }, constraint::PointMassFormConstraint, distribution)
    # returns a tuple of `lower` and `upper` boundaries
    return (-Inf, Inf)
end
```
"""
struct PointMassFormConstraint{F, P, B} <: AbstractFormConstraint
    optimizer      :: F
    starting_point :: P
    boundaries     :: B
end

Base.show(io::IO, ::PointMassFormConstraint) = print(io, "PointMassFormConstraint()")

PointMassFormConstraint(;
    optimizer = default_point_mass_form_constraint_optimizer,
    starting_point = default_point_mass_form_constraint_starting_point,
    boundaries = default_point_mass_form_constraint_boundaries
) = PointMassFormConstraint(optimizer, starting_point, boundaries)

ReactiveMP.default_form_check_strategy(::PointMassFormConstraint) = FormConstraintCheckLast()

ReactiveMP.default_prod_constraint(::PointMassFormConstraint) = GenericProd()

call_optimizer(pmconstraint::PointMassFormConstraint, distribution::D) where {D}      = pmconstraint.optimizer(variate_form(D), value_support(D), pmconstraint, distribution)
call_boundaries(pmconstraint::PointMassFormConstraint, distribution::D) where {D}     = pmconstraint.boundaries(variate_form(D), value_support(D), pmconstraint, distribution)
call_starting_point(pmconstraint::PointMassFormConstraint, distribution::D) where {D} = pmconstraint.starting_point(variate_form(D), value_support(D), pmconstraint, distribution)

ReactiveMP.constrain_form(pmconstraint::PointMassFormConstraint, distribution) = call_optimizer(pmconstraint, distribution)

# There is no need to call the optimizer on a `Distribution` object since they should have a well defined `mode`
ReactiveMP.constrain_form(::PointMassFormConstraint, distribution::Distribution) = PointMass(mode(distribution))

# Categorical distribution has an exception since `mode` does not return a one-hot vector, which is required for backwards compatibility with `Categorical` marginals
ReactiveMP.constrain_form(::PointMassFormConstraint, distribution::DiscreteNonParametric{T, P, Ts, Ps}) where {T, P, Ts, Ps} = begin
    pv = probvec(distribution)
    result = SparseVector{P, Int64}(undef, 5)
    result[argmax(pv)] = one(P)
    PointMass(result)
end

"""
    default_point_mass_form_constraint_optimizer(::Type{<:VariateType}, ::Type{<:ValueSupport}, constraint::PointMassFormConstraint, distribution)

Defines a default optimisation procedure for the `PointMassFormConstraint`. By default uses `Optim.jl` package to find argmin of `-logpdf(x)`.
Uses the `starting_point` and `boundaries` callbacks to determine the starting point and boundaries for the optimisation procedure.
"""
function default_point_mass_form_constraint_optimizer end

function default_point_mass_form_constraint_optimizer(::Type{Univariate}, ::Type{Continuous}, constraint::PointMassFormConstraint, distribution)
    target = let distribution = distribution
        (x) -> -logpdf(distribution, x[1])
    end

    lower, upper = call_boundaries(constraint, distribution)

    result = if isinf(lower) && isinf(upper)
        Optim.optimize(target, call_starting_point(constraint, distribution), Optim.LBFGS())
    else
        Optim.optimize(target, [lower], [upper], call_starting_point(constraint, distribution), Optim.Fminbox(Optim.GradientDescent()))
    end

    if Optim.converged(result)
        return PointMass(Optim.minimizer(result)[1])
    else
        error("Optimisation procedure for point mass estimation did not converge", result)
    end
end

function default_point_mass_form_constraint_optimizer(::Type{Univariate}, ::Type{Discrete}, constraint::PointMassFormConstraint, distribution)

    # fetch probvec
    p = probvec(distribution)

    # create new probvec
    p_new = zeros(length(p))
    p_new[argmax(p)] = 1

    return PointMass(p_new)
end

"""
    default_point_mass_form_constraint_boundaries(::Type{<:VariateType}, ::Type{<:ValueSupport}, constraint::PointMassFormConstraint, distribution)

Defines a default boundaries for the `PointMassFormConstraint`. By default simply uses the support of the distribution.
"""
function default_point_mass_form_constraint_boundaries end

function default_point_mass_form_constraint_boundaries(::Type{Univariate}, ::Type{Continuous}, constraint::PointMassFormConstraint, distribution)
    return __default_univariate_boundaries(support(distribution))
end

__default_univariate_boundaries(interval::AbstractRange) = (minimum(interval), maximum(interval))
__default_univariate_boundaries(interval::Distributions.RealInterval) = (minimum(interval), maximum(interval))
__default_univariate_boundaries(domain::Domain) = (infimum(domain), supremum(domain))

"""
    default_point_mass_form_constraint_starting_point(::Type{<:VariateType}, ::Type{<:ValueSupport}, constraint::PointMassFormConstraint, distribution)

Defines a default starting point for the `PointMassFormConstraint`. By default uses the support of the distribution.
If support is unbounded returns a zero point. Otherwise throws an error.
"""
function default_point_mass_form_constraint_starting_point end

function default_point_mass_form_constraint_starting_point(::Type{Univariate}, ::Type{Continuous}, constraint::PointMassFormConstraint, distribution)
    lower, upper = call_boundaries(constraint, distribution)
    return if isinf(lower) && isinf(upper)
        return zeros(1)
    else
        error("No default starting point specified for a range [ $(lower), $(upper) ]")
    end
end
