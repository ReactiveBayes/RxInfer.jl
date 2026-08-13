export SampleListFormConstraint, LeftProposal, RightProposal, AutoProposal

import ReactiveMP:
    default_form_check_strategy, default_prod_constraint, constrain_form
import BayesBase: AbstractContinuousGenericLogPdf, BootstrapImportanceSampling

using Random

"""
Uses the left argument in the `prod` call as the proposal distribution in the SampleList approximation.
"""
struct LeftProposal end

"""
Uses the right argument in the `prod` call as the proposal distribution in the SampleList approximation.
"""
struct RightProposal end

"""
Tries to determine the proposal distribution in the SampleList approximation automatically.
"""
struct AutoProposal end

"""
    SampleListFormConstraint([rng], nsamples, strategy, method)

One of the form constraint objects. Approximates `ProductOf` with a `SampleList` object.

# Arguments

- `rng`: (optional) random number generator used to draw the samples. Defaults to
  `Random.default_rng()`. Pass an explicit, seeded generator (e.g. `MersenneTwister(42)`
  or a `StableRNG`) to obtain reproducible sample-list approximations.
- `nsamples`: number of samples in the resulting `SampleList`.
- `strategy`: proposal selection strategy, one of [`AutoProposal`](@ref), [`LeftProposal`](@ref)
  or [`RightProposal`](@ref). Defaults to `AutoProposal()`.
- `method`: sampling method. Defaults to `BootstrapImportanceSampling()`.

!!! note
    When no `rng` is provided the constraint uses `Random.default_rng()`, so the drawn
    samples depend on the global RNG state. For reproducible inference either seed the
    global RNG with `Random.seed!` or pass an explicit `rng` as the first argument.
"""
struct SampleListFormConstraint{N, R, S, M} <: AbstractFormConstraint
    rng      :: R
    strategy :: S
    method   :: M
end

Base.show(io::IO, constraint::SampleListFormConstraint) = print(
    io,
    "SampleListFormConstraint(",
    constraint.rng,
    ", ",
    constraint.strategy,
    ", ",
    constraint.method,
    ")",
)

SampleListFormConstraint(nsamples::Int, strategy::S = AutoProposal(), method::M = BootstrapImportanceSampling()) where {S, M}                           = SampleListFormConstraint(Random.default_rng(), nsamples, strategy, method)
SampleListFormConstraint(rng::R, nsamples::Int, strategy::S = AutoProposal(), method::M = BootstrapImportanceSampling()) where {R <: AbstractRNG, S, M} = SampleListFormConstraint{nsamples, R, S, M}(rng, strategy, method)

ReactiveMP.default_form_check_strategy(::SampleListFormConstraint) =
    FormConstraintCheckLast()

ReactiveMP.default_prod_constraint(::SampleListFormConstraint) = GenericProd()

__approximate(constraint::SampleListFormConstraint{N, R, S, M}, left, right) where {N, R, S <: LeftProposal, M}  = BayesBase.approximate_prod_with_sample_list(constraint.rng, constraint.method, left, right, N)
__approximate(constraint::SampleListFormConstraint{N, R, S, M}, left, right) where {N, R, S <: RightProposal, M} = BayesBase.approximate_prod_with_sample_list(constraint.rng, constraint.method, right, left, N)

# The logic here is that the `__approximate` function will try to pick as a proposal candidate an object
# which is not in the `AutoProposalLowPriorityCandidates` list
# For example if we have a product of a `Gaussian` and a `ContinuousGenericLogPdf` the `AutoProposal` strategy
# should pick the `Gaussian` as the proposal distribution
const AutoProposalLowPriorityCandidates = Union{
    AbstractContinuousGenericLogPdf, LinearizedProductOf
}

function __approximate(
    constraint::SampleListFormConstraint{N, R, S, M},
    left::AutoProposalLowPriorityCandidates,
    right,
) where {N, R, S <: AutoProposal, M}
    return BayesBase.approximate_prod_with_sample_list(
        constraint.rng, constraint.method, right, left, N
    )
end

function __approximate(
    constraint::SampleListFormConstraint{N, R, S, M},
    left,
    right::AutoProposalLowPriorityCandidates,
) where {N, R, S <: AutoProposal, M}
    return BayesBase.approximate_prod_with_sample_list(
        constraint.rng, constraint.method, left, right, N
    )
end

function __approximate(
    constraint::SampleListFormConstraint{N, R, S, M},
    left::AutoProposalLowPriorityCandidates,
    right::AutoProposalLowPriorityCandidates,
) where {N, R, S <: AutoProposal, M}
    return error(
        "Cannot approximate the product of $(left) and $(right) as a sample list. The `AutoProposal` strategy cannot choose a proposal distribution. Use either `LeftProposal` or `RightProposal` in the sample list form constraint specification.",
    )
end

function ReactiveMP.constrain_form(::SampleListFormConstraint, something)
    return something
end

function ReactiveMP.constrain_form(
    constraint::SampleListFormConstraint, product::ProductOf
)
    left  = ReactiveMP.constrain_form(constraint, BayesBase.getleft(product))
    right = ReactiveMP.constrain_form(constraint, BayesBase.getright(product))
    return __approximate(constraint, left, right)
end
