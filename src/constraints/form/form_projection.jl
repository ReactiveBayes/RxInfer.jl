import DeltaMessagePassingRules: LaplaceForm, laplace_form, VariationalForm, variational_form, MomentForm, moment_form

# DeltaMessagePassingRules' `LaplaceForm`, `VariationalForm` and `MomentForm` as form constraints.
# Their projections, `laplace_form`, `variational_form` and `moment_form`, report the log of the
# product's normalizer (Laplace's approximation of it, the evidence lower bound, or the moments'
# quadrature), which completes the log scale of a product whose normalizer was unknown. Each
# variable's context keeps the last mode, where the next projection starts.

const ProjectionForm = Union{LaplaceForm, VariationalForm, MomentForm}

ReactiveMP.default_form_check_strategy(::ProjectionForm) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::ProjectionForm) = GenericProd()

mutable struct ProjectionFormContext
    previous::Any
    # how the last projection went, for the approximation report
    last::Any
end

ReactiveMP.prepare_context(::ProjectionForm) = ProjectionFormContext(nothing, nothing)

project(form::LaplaceForm, density, previous) = laplace_form(form, density, previous)
project(form::VariationalForm, density, previous) = variational_form(form, density, previous)
project(form::MomentForm, density, previous) = moment_form(form, density, previous)

function ReactiveMP.constrain_form_logscale(form::ProjectionForm, context::ProjectionFormContext, density)
    result = project(form, density, context.previous)
    isnothing(result.logZ) || (context.previous = result.mode)
    context.last = (; result.method, result.logZ, result.converged, result.iterations)
    return result.distribution, result.logZ
end

form_outcome(::ProjectionForm, context::ProjectionFormContext, distribution, result) = context.last

ReactiveMP.constrain_form(form::ProjectionForm, context::ProjectionFormContext, density) =
    first(ReactiveMP.constrain_form_logscale(form, context, density))
