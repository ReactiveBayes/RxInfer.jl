using Rocket

import Base: show, setindex!

mutable struct ScoreActor{L} <: Rocket.Actor{L}
    score  :: Matrix{L}
    cframe :: Int
    cindex :: Int
    valid  :: BitVector
    # Matrix height is capacity; a converged event may fill only a prefix of its column.
    counts::Vector{Int}
end

ScoreActor(iterations::Int, keep::Int = 1) = ScoreActor(Real, iterations, keep)
ScoreActor(::Type{L}, iterations::Int, keep::Int = 1) where {L <: Real} =
    ScoreActor{L}(
        zeros(L, iterations, keep), 1, 0, falses(keep), zeros(Int, keep)
    )

function valid_score_frames(actor::ScoreActor)
    # Ring storage order differs from observation order after wrapping.
    return filter(
        i -> actor.valid[i],
        vcat((actor.cframe + 1):getnframes(actor), 1:actor.cframe),
    )
end

Base.show(io::IO, ::ScoreActor{L}) where {L} = print(io, "ScoreActor(", L, ")")
Base.setindex!(actor::ScoreActor, data, frame, index) =
    actor.score[index, frame] = data

function getvalid(actor::ScoreActor)
    return Iterators.flatten(
        view(actor.score, 1:actor.counts[i], i) for
        i in valid_score_frames(actor)
    )
end

getniterations(actor::ScoreActor) = size(actor.score, 1)
getnframes(actor::ScoreActor)     = size(actor.score, 2)

function Rocket.on_next!(actor::ScoreActor{L}, data::L) where {L}
    iterations = getniterations(actor)
    nframes    = getnframes(actor)

    # Obtain current `frame` and data `index` position
    cframe = actor.cframe
    cindex = actor.cindex + 1

    # A released partial column is finished even when its capacity was not reached.
    if cindex > iterations || actor.valid[cframe]
        # We also check that the previous frame has been released
        @assert actor.valid[cframe] "Broken `ScoreActor` state, previous frame has not been released"
        cframe = ifelse(cframe + 1 > nframes, 1, cframe + 1)
        cindex = 1
        actor.valid[cframe] = false
    end

    actor[cframe, cindex] = data

    actor.cindex = cindex
    actor.cframe = cframe

    return nothing
end

function Rocket.on_error!(actor::ScoreActor, err)
    error(err)
end

function Rocket.on_complete!(actor::ScoreActor)
    Rocket.release!(actor)
    nothing
end

function Rocket.release!(actor::ScoreActor, warn = true; allow_partial = false)
    iterations = getniterations(actor)
    cframe     = actor.cframe
    cindex     = actor.cindex
    allow_partial && cindex == 0 && return nothing

    if warn && (cindex !== iterations) && !allow_partial
        @warn "Invalid `release!` call on `ScoreActor`. The current frame has not been fully specified"
    else
        @assert !actor.valid[cframe] "Broken `ScoreActor` state, cannot `release!` a valid frame of free energy values"
        actor.valid[cframe] = true
        actor.counts[cframe] = cindex
    end

    return nothing
end

function score_snapshot(actor::ScoreActor)
    return collect(getvalid(actor))
end

function score_snapshot_final(actor::ScoreActor)
    return [actor.score[actor.counts[i], i] for i in valid_score_frames(actor)]
end

function score_snapshot_iterations(actor::ScoreActor, slice = nothing)
    frames = valid_score_frames(actor)
    niterations = isempty(frames) ? 0 : maximum(actor.counts[i] for i in frames)
    # Average only events that reached iteration k; unused cells are not zero BFE samples.
    result = [
        sum(actor.score[k, i] for i in frames if actor.counts[i] >= k) /
        count(i -> actor.counts[i] >= k, frames) for k in 1:niterations
    ]

    return slice_snapshot(result, slice)
end

slice_snapshot(vector, ::Nothing) = vector
slice_snapshot(vector, range::AbstractRange) = vector[range]
slice_snapshot(vector, count::Int) =
    if length(vector) === count
        vector
    else
        vector[firstindex(vector):(firstindex(vector) + count - 1)]
    end
