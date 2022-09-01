# NamedTuple helpers

fields(::NamedTuple{F})            where {F} = F
fields(::Type{ <: NamedTuple{F} }) where {F} = F

nthasfield(field::Symbol, ntuple::NamedTuple)            = field ∈ fields(ntuple)
nthasfield(field::Symbol, ntuple::Type{ <: NamedTuple }) = field ∈ fields(ntuple)

# Reduce helpers

sumreduce(array) = reduce(+, array)
