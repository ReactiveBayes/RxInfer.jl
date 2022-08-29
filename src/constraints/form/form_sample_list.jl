export LeftProposal, RightProposal

import ReactiveMP: is_point_mass_form_constraint, default_form_check_strategy, default_prod_constraint, make_form_constraint, constrain_form

using Random

struct LeftProposal end
struct RightProposal end
struct AutoProposal end

"""
    SampleListFormConstraint(rng, strategy, method)

One of the form constraint objects. Approximates `DistProduct` with a SampleList object. 

# Traits 
- `is_point_mass_form_constraint` = `false`
- `default_form_check_strategy`   = `FormConstraintCheckLast()`
- `default_prod_constraint`       = `ProdGeneric()`
- `make_form_constraint`          = `SampleList` (for use in `@constraints` macro)

See also: [`constrain_form`](@ref), [`DistProduct`](@ref)
"""
struct SampleListFormConstraint{N, R, S, M} <: AbstractFormConstraint
    rng      :: R
    strategy :: S
    method   :: M
end

Base.show(io::IO, constraint::SampleListFormConstraint) =
    print(io, "SampleListFormConstraint(", constraint.rng, ", ", constraint.strategy, ", ", constraint.method, ")")

SampleListFormConstraint(nsamples::Int, strategy::S = AutoProposal(), method::M = ReactiveMP.BootstrapImportanceSampling()) where {S, M}                           = SampleListFormConstraint(Random.GLOBAL_RNG, nsamples, strategy, method)
SampleListFormConstraint(rng::R, nsamples::Int, strategy::S = AutoProposal(), method::M = ReactiveMP.BootstrapImportanceSampling()) where {R <: AbstractRNG, S, M} = SampleListFormConstraint{nsamples, R, S, M}(rng, strategy, method)

ReactiveMP.is_point_mass_form_constraint(::SampleListFormConstraint) = false

ReactiveMP.default_form_check_strategy(::SampleListFormConstraint) = FormConstraintCheckLast()

ReactiveMP.default_prod_constraint(::SampleListFormConstraint) = ProdGeneric()

ReactiveMP.make_form_constraint(::Type{SampleList}, args...; kwargs...) = SampleListFormConstraint(args...; kwargs...)

__approximate(constraint::SampleListFormConstraint{N, R, S, M}, left, right) where {N, R, S <: LeftProposal, M}  = approximate_prod_with_sample_list(constraint.rng, constraint.method, left, right, N)
__approximate(constraint::SampleListFormConstraint{N, R, S, M}, left, right) where {N, R, S <: RightProposal, M} = approximate_prod_with_sample_list(constraint.rng, constraint.method, right, left, N)

function __approximate(constraint::SampleListFormConstraint{N, R, S, M}, left::ContinuousUnivariateLogPdf, right) where {N, R, S <: AutoProposal, M}
    return approximate_prod_with_sample_list(constraint.rng, constraint.method, right, left, N)
end

function __approximate(constraint::SampleListFormConstraint{N, R, S, M}, left, right::ContinuousUnivariateLogPdf) where {N, R, S <: AutoProposal, M}
    return approximate_prod_with_sample_list(constraint.rng, constraint.method, left, right, N)
end

function __approximate(constraint::SampleListFormConstraint{N, R, S, M}, left::ContinuousUnivariateLogPdf, right::ContinuousUnivariateLogPdf) where {N, R, S <: AutoProposal, M} 
    return error("Cannot approximate the product of $(left) and $(right) as a sample list.")
end

function ReactiveMP.constrain_form(::SampleListFormConstraint, something)
    return something
end

function ReactiveMP.constrain_form(constraint::SampleListFormConstraint, product::DistProduct)
    left  = ReactiveMP.constrain_form(constraint, getleft(product))
    right = ReactiveMP.constrain_form(constraint, getright(product))
    return __approximate(constraint, left, right)
end
