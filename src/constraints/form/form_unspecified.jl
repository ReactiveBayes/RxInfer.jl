export UnspecifiedFormConstraint



"""
    UnspecifiedFormConstraint

One of the form constraint objects. Does not imply any form constraints and simply returns the same object as receives.
However it does not allow `DistProduct` to be a valid functional form in the inference backend.

# Traits 
- `is_point_mass_form_constraint` = `false`
- `default_form_check_strategy`   = `FormConstraintCheckLast()`
- `default_prod_constraint`       = `ProdAnalytical()`
- `make_form_constraint`          = `Nothing` (for use in `@constraints` macro)

See also: [`constrain_form`](@ref), [`DistProduct`](@ref)
"""
struct UnspecifiedFormConstraint <: AbstractFormConstraint end

ReactiveMP.is_point_mass_form_constraint(::UnspecifiedFormConstraint) = false

ReactiveMP.default_form_check_strategy(::UnspecifiedFormConstraint) = FormConstraintCheckLast()

ReactiveMP.default_prod_constraint(::UnspecifiedFormConstraint) = ProdAnalytical()

ReactiveMP.make_form_constraint(::Type{<:Nothing}) = UnspecifiedFormConstraint()

ReactiveMP.constrain_form(::UnspecifiedFormConstraint, something)              = something
ReactiveMP.constrain_form(::UnspecifiedFormConstraint, something::DistProduct) = error("`DistProduct` object cannot be used as a functional form in inference backend. Use form constraints to restrict the functional form of marginal posteriors.")
