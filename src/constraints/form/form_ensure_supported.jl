import ReactiveMP: AbstractFormConstraint

# This is an internal functional form constraint that only checks that the result 
# is of a supported form. Displays a user-friendly error message if the form is not supported.
struct EnsureSupportedFunctionalForm <: AbstractFormConstraint
    prefix::Symbol
    name::Symbol
    index::Any
end

ReactiveMP.default_form_check_strategy(::EnsureSupportedFunctionalForm) = FormConstraintCheckLast()

ReactiveMP.default_prod_constraint(::EnsureSupportedFunctionalForm) = GenericProd()

function ReactiveMP.constrain_form(constraint::EnsureSupportedFunctionalForm, something)
    if typeof(something) <: ProductOf || typeof(something) <: LinearizedProductOf
        expr = string(constraint.prefix, '(', constraint.name, isnothing(constraint.index) ? "" : string('[', constraint.index, ']'), ')')
        expr_noindex = string(constraint.prefix, '(', constraint.name, ')')
        error(lazy"""
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
        """)
    end
    return something
end
