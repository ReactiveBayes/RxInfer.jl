# NamedTuple helpers

fields(::NamedTuple{F}) where {F} = F
hasfield(field::Symbol, ntuple::NamedTuple) = field ∈ fields(ntuple)

# Reduce helpers

sumreduce(array) = reduce(+, array)