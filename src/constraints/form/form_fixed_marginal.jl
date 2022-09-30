
import ReactiveMP: is_point_mass_form_constraint, default_form_check_strategy, default_prod_constraint, make_form_constraint, constrain_form

"""
    FixedMarginalFormConstraint

One of the form constraint objects. Provides a constraint on the marginal distribution such that it remains fixed during inference. 
Can be viewed as blocking of updates of a specific edge associated with the marginal. If `nothing` is passed then the computed posterior marginal is returned.

# Traits 
- `is_point_mass_form_constraint` = `false`
- `default_form_check_strategy`   = `FormConstraintCheckLast()`
- `default_prod_constraint`       = `ProdAnalytical()`
- `make_form_constraint`          = `Marginal` (for use in `@constraints` macro)

See also: [`ReactiveMP.constrain_form`](@ref), [`ReactiveMP.DistProduct`](@ref)
"""
mutable struct FixedMarginalFormConstraint <: ReactiveMP.AbstractFormConstraint
    fixed_value::Any
end

ReactiveMP.is_point_mass_form_constraint(::FixedMarginalFormConstraint) = false

ReactiveMP.default_form_check_strategy(::FixedMarginalFormConstraint) = FormConstraintCheckLast()

ReactiveMP.default_prod_constraint(::FixedMarginalFormConstraint) = ProdGeneric()

ReactiveMP.make_form_constraint(::Type{<:Marginal}, fixed_value) = FixedMarginalFormConstraint(fixed_value)

ReactiveMP.constrain_form(constraint::FixedMarginalFormConstraint, something) = constraint.fixed_value !== nothing ? constraint.fixed_value : something
