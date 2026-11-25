import ReactiveMP: AbstractFormConstraint

# This is an internal functional form constraint that only checks that the result
# is of a supported form. Displays a user-friendly error message if the form is not supported.
struct EnsureSupportedFunctionalForm <: AbstractFormConstraint
    prefix::Symbol
    name::Symbol
    index::Any
    # whether the form before this check leaves a product it cannot project as it is, on
    # purpose (`nomode = :leave`), for the rule fallback of the next node to take
    leaves_products::Bool
end

EnsureSupportedFunctionalForm(prefix::Symbol, name::Symbol, index) =
    EnsureSupportedFunctionalForm(prefix, name, index, false)

# Whether a form constraint leaves the products it cannot project unprojected
leaves_products(form) = false
leaves_products(form::ReactiveMP.WrappedFormConstraint) = leaves_products(form.constraint)
leaves_products(form::CompositeFormConstraint) = any(leaves_products, form.constraints)
leaves_products(form::Union{DeltaMessagePassingRules.LaplaceForm, DeltaMessagePassingRules.MomentForm, DeltaMessagePassingRules.VariationalForm}) =
    form.nomode === :leave

ReactiveMP.default_form_check_strategy(::EnsureSupportedFunctionalForm) =
    FormConstraintCheckLast()

ReactiveMP.default_prod_constraint(::EnsureSupportedFunctionalForm) =
    GenericProd()

function ReactiveMP.constrain_form(
        constraint::EnsureSupportedFunctionalForm, something
    )
    if !constraint.leaves_products && (
            typeof(something) <: ProductOf ||
                typeof(something) <: LinearizedProductOf
        )
        expr = string(
            constraint.prefix,
            '(',
            constraint.name,
            if isnothing(constraint.index)
                ""
            else
                string('[', constraint.index, ']')
            end,
            ')',
        )
        expr_noindex = string(constraint.prefix, '(', constraint.name, ')')
        error(
            lazy"""
            The expression `$expr` has an undefined functional form of type `$(typeof(something))`. 
            This is likely because the inference backend does not support the product of these distributions. 
            As a result, `RxInfer` cannot compute key quantities such as the `mean` or `var` of `$expr`.

            Possible solutions:
            - Alter model specification to ensure the prior is conjugate (see https://en.wikipedia.org/wiki/Conjugate_prior).
            - Implement the `BayesBase.prod` method (refer to the `BayesBase` documentation for guidance).
            - Use a functional form constraint to specify the posterior form with the `@constraints` macro. For example:
            ```julia
            using ExponentialFamilyProjection

            @constraints begin
                $(expr_noindex) :: ProjectedTo(NormalMeanVariance)
            end
            ```
            Refer to the documentation for more details on functional form constraints.
            """,
        )
    end
    return something
end
