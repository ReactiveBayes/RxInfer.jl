import DeltaMessagePassingRules: LaplaceForm, laplace_form, VariationalForm, variational_form

# DeltaMessagePassingRules' `LaplaceForm` and `VariationalForm` as form constraints: their
# projections, `laplace_form` and `variational_form`, report the log of the product's normaliser
# (Laplace's evidence approximation, or the evidence lower bound), which completes the log scale of
# a product whose normaliser was unknown, and each variable's context keeps the last point, where
# the next projection starts.

const ProjectionForm = Union{LaplaceForm, VariationalForm}

ReactiveMP.default_form_check_strategy(::ProjectionForm) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::ProjectionForm) = GenericProd()

mutable struct LaplaceFormContext
    previous::Any
    # how the last projection went, for the approximation report
    last::Any
end

ReactiveMP.prepare_context(::ProjectionForm) = LaplaceFormContext(nothing, nothing)

project(form::LaplaceForm, density, previous) = laplace_form(form, density, previous)
project(form::VariationalForm, density, previous) = variational_form(form, density, previous)

function ReactiveMP.constrain_form_logscale(form::ProjectionForm, context::LaplaceFormContext, density)
    result = project(form, density, context.previous)
    isnothing(result.logZ) || (context.previous = result.mode)
    context.last = (; result.method, result.logZ, result.converged, result.iterations)
    return result.distribution, result.logZ
end

form_outcome(::ProjectionForm, context::LaplaceFormContext, distribution, result) = context.last

ReactiveMP.constrain_form(form::ProjectionForm, context::LaplaceFormContext, density) =
    first(ReactiveMP.constrain_form_logscale(form, context, density))
