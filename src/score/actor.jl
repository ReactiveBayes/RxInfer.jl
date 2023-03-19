using Rocket

import Base: show, setindex!

mutable struct ScoreActor{L} <: Rocket.Actor{L}
    score  :: Matrix{L}
    cframe :: Int
    cindex :: Int
    valid  :: BitVector
end

ScoreActor(iterations::Int, keep::Int = 1) = ScoreActor(Real, iterations, keep)
ScoreActor(::Type{L}, iterations::Int, keep::Int = 1) where {L <: Real} = ScoreActor{L}(Matrix{L}(undef, iterations, keep), 1, 0, falses(keep))

Base.show(io::IO, ::ScoreActor{L}) where {L} = print(io, "ScoreActor(", L, ")")
Base.setindex!(actor::ScoreActor, data, frame, index) = actor.score[index, frame] = data

function getvalid(actor::ScoreActor)
    firstframe  = something(findnext(actor.valid, actor.cframe + 1), 1)
    continuous  = Iterators.flatten(eachcol(actor.score))
    niterations = getniterations(actor)
    ndrop       = (firstframe - 1) * niterations
    ntotal      = sum(actor.valid) * niterations

    return Iterators.take(Iterators.drop(Iterators.cycle(continuous), ndrop), ntotal)
end

getniterations(actor::ScoreActor) = size(actor.score, 1)
getnframes(actor::ScoreActor)     = size(actor.score, 2)

function Rocket.on_next!(actor::ScoreActor{L}, data::L) where {L}
    iterations = getniterations(actor)
    nframes    = getnframes(actor)

    # Obtain current `frame` and data `index` position
    cframe = actor.cframe
    cindex = actor.cindex + 1

    # If `cindex` overflows number of iterations it means we have to save 
    # our data in the next frame 
    # This functionality is also present in the `release!` function
    if cindex > iterations
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

function Rocket.release!(actor::ScoreActor)
    iterations = getniterations(actor)
    cframe     = actor.cframe
    cindex     = actor.cindex

    if cindex !== iterations
        @warn "Invalid `release!` call on `ScoreActor`. The current frame has not been fully specified"
    else
        @assert !actor.valid[cframe] "Broken `ScoreActor` state, cannot `release!` a valid frame of free energy values"
        actor.valid[cframe] = true
    end

    return nothing
end

function score_snapshot(actor::ScoreActor)
    return collect(getvalid(actor))
end

function score_snapshot_final(actor::ScoreActor)
    iters = getniterations(actor)
    return score_snapshot(actor)[iters:iters:end]
end

function score_snapshot_iterations(actor::ScoreActor)
    result = vec(sum(view(actor.score, :, actor.valid), dims = 2))

    map!(Base.Fix2(/, sum(actor.valid)), result, result)

    return result
end

#if allow failed, there are some #undef in fe_values. This function aims to removing these #undef.
function score_snapshot_when_interupt(actor::ScoreActor)
    result=actor.score
    bb=filter(i -> isassigned(result, i), 1:length(result))

    result=result[bb]

    return result
end