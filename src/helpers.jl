# NamedTuple helpers

fields(::NamedTuple{F}) where {F}         = F
fields(::Type{<:NamedTuple{F}}) where {F} = F

nthasfield(field::Symbol, ntuple::NamedTuple)         = field ∈ fields(ntuple)
nthasfield(field::Symbol, ntuple::Type{<:NamedTuple}) = field ∈ fields(ntuple)

# Bool helpers

ensure_bool_or_nothing(value::Bool) = value
ensure_bool_or_nothing(anything_else) = nothing

# Tuple helpers

as_tuple(something)    = (something,)
as_tuple(tuple::Tuple) = tuple

# Val helpers 

unval(::Val{X}) where {X} = X
unval(something) = error("Cannot un-val, the value is not `Val`")

# Reduce helpers

sumreduce(array) = reduce(+, array)
