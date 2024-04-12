export FixedMarginalFormConstraint

import ReactiveMP: default_form_check_strategy, default_prod_constraint, constrain_form

"""
    FixedMarginalFormConstraint

One of the form constraint objects. Provides a constraint on the marginal distribution such that it remains fixed during inference. 
Can be viewed as blocking of updates of a specific edge associated with the marginal. If `nothing` is passed then the computed posterior marginal is returned.
Use `.fixed_value` field to update the value of the constraint.
"""
mutable struct FixedMarginalFormConstraint <: ReactiveMP.AbstractFormConstraint
    fixed_value::Any
end

ReactiveMP.default_form_check_strategy(::FixedMarginalFormConstraint) = FormConstraintCheckLast()

ReactiveMP.default_prod_constraint(::FixedMarginalFormConstraint) = GenericProd()

ReactiveMP.constrain_form(constraint::FixedMarginalFormConstraint, something) = constraint.fixed_value !== nothing ? constraint.fixed_value : something
