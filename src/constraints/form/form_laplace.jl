import DeltaMessagePassingRules: LaplaceForm, laplace_form

# DeltaMessagePassingRules' `LaplaceForm` as a form constraint: its projection, `laplace_form`,
# reports the log of the product's normaliser, which completes the log scale of a product whose
# normaliser was unknown, and each variable's context keeps the last mode, where the next
# projection starts.

ReactiveMP.default_form_check_strategy(::LaplaceForm) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::LaplaceForm) = GenericProd()

mutable struct LaplaceFormContext
    previous::Any
    # how the last projection went, for the approximation report
    last::Any
end

ReactiveMP.prepare_context(::LaplaceForm) = LaplaceFormContext(nothing, nothing)

function ReactiveMP.constrain_form_logscale(form::LaplaceForm, context::LaplaceFormContext, density)
    result = laplace_form(form, density, context.previous)
    isnothing(result.logZ) || (context.previous = result.mode)
    context.last = (; result.method, result.logZ, result.converged, result.iterations)
    return result.distribution, result.logZ
end

form_outcome(::LaplaceForm, context::LaplaceFormContext, distribution, result) = context.last

ReactiveMP.constrain_form(form::LaplaceForm, context::LaplaceFormContext, density) =
    first(ReactiveMP.constrain_form_logscale(form, context, density))
